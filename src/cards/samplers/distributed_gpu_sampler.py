r"""Distributed GPU MCMC sampling backbone."""

import hashlib
import logging
from pathlib import Path
from typing import cast

import cupy as cp
import h5py
import torch
from mpi4py import MPI

from cards.io.io_manager import IOManager
from cards.models.base_model import BaseDistributedModel
from cards.samplers.base_sampler import BaseSampler, SamplerParameters


class DistributedGpuSampler(BaseSampler):
    r"""MPI-based distributed GPU implementation of the MCMC sampling backbone.

    This class orchestrates multi-GPU MCMC chains using `mpi4py`and CuPy. It derives
    statistically independent PyTorch CUDA random streams per worker via deterministic
    SHA-256 seed hashing, and utilizes parallel HDF5 (`mpio`) for collective and
    efficient checkpoint I/O.
    """

    model: BaseDistributedModel
    rng: torch.Generator

    def __init__(
        self,
        comm: MPI.Comm,
        params: SamplerParameters,
        model: BaseDistributedModel,
        logger: logging.Logger | None = None,
    ) -> None:
        self.comm = comm
        super().__init__(params, model, logger)

        self._start_gpu = cp.cuda.Event()
        self._end_gpu = cp.cuda.Event()

    def _setup_rank(self) -> int:
        return self.comm.Get_rank()

    def _setup_rng(self) -> torch.Generator:
        r"""Derive an independent worker seed via SHA-256 hashing of (rank, master_seed)."""
        combined = f"{self.rank}{self.seed}"
        self._local_seed = int(hashlib.sha256(combined.encode()).hexdigest(), 16) % (
            2**32
        )
        return torch.Generator(device="cuda").manual_seed(self._local_seed)

    def _setup_io_manager(self) -> IOManager:
        return IOManager(
            self.ckpt_size,
            self.model.local_sizes,
            self.model.global_sizes,
            self.model.slices,
        )

    def _start_timer(self) -> None:
        self._start_gpu.record()

    def _stop_timer(self) -> None:
        r"""Record the GPU stop-event and force a blocking CUDA stream synchronization."""
        self._end_gpu.record()
        self._end_gpu.synchronize()

    def _get_elapsed_time(self) -> float:
        return cp.cuda.get_elapsed_time(self._start_gpu, self._end_gpu) * 1e-3

    def _get_potential(self) -> float:
        r"""Compute local GPU potential and execute a blocking collective `MPI.SUM`
        reduction."""
        local_potential = self.model.compute_potential()
        return cast(float, self.comm.reduce(local_potential, MPI.SUM, root=0))

    def _save_checkpoint(self, ckpt_path: Path) -> None:
        r"""Trigger a collective parallel HDF5 write across all MPI processes."""
        with h5py.File(ckpt_path, "w", driver="mpio", comm=self.comm) as file:
            self.io_manager.save_dict(
                self.model.get_states(),
                file,
                self.model.global_sizes,
                self.model.slices,
            )

            self.io_manager.save_dict(
                self.model.get_estimates(),
                file,
                self.model.global_sizes,
                self.model.slices,
            )

            self.io_manager.save_rng_torch(
                self.rng, self._local_seed, file, self.rank, self.comm.Get_size()
            )

            self.io_manager.save_thread_array(
                self._computation_time,
                self.rank,
                self.comm.Get_size(),
                "computation_time",
                file,
            )

            if self.rank == 0:
                self.io_manager.save_local_array(self._potential, "potential", file)

    def _load_checkpoint(self) -> None:
        r"""Load sampler state collaboratively from disk via parallel HDF5 (`mpio`)."""
        with h5py.File(self.restart_path, "r", driver="mpio", comm=self.comm) as file:
            self.model.set_states(self.io_manager.load_states(file, self.model.vars))
            self.io_manager.load_rng_torch(self.rng, file, self.rank)
