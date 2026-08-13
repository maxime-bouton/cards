r"""Distributed GPU MCMC sampling backbone."""

import logging
from pathlib import Path
from typing import cast

import cupy as cp
import torch
from mpi4py import MPI

from cards.io.io_manager import IOManager
from cards.models.base_model import BaseDistributedModel
from cards.samplers.sampler import Sampler, SamplerParameters


class DistributedGpuSampler(Sampler):
    r"""MPI-based distributed GPU implementation of the MCMC sampling backbone.

    This class orchestrates multi-GPU MCMC chains using :mod:`mpi4py` and :mod:`CuPy`.
    It derives statistically independent :mod:`torch` CUDA random streams per worker via
    deterministic SHA-256 seed hashing, and utilizes parallel HDF5 (``mpio``) for
    collective and efficient checkpoint I/O.
    """

    model: BaseDistributedModel

    def __init__(
        self,
        io_mng: IOManager,
        comm: MPI.Comm,
        params: SamplerParameters,
        model: BaseDistributedModel,
        rng: torch.Generator,
        logger: logging.Logger | None = None,
    ) -> None:
        self.comm = comm
        super().__init__(io_mng, params, model, rng, logger)

        self._start_gpu = cp.cuda.Event()
        self._end_gpu = cp.cuda.Event()

    def _setup_rank(self) -> int:
        return self.comm.Get_rank()

    def _start_timer(self) -> None:
        self._start_gpu.record()

    def _stop_timer(self) -> None:
        r"""Record the GPU stop-event and force a blocking CUDA stream synchronization."""
        self._end_gpu.record()
        self._end_gpu.synchronize()

    def _get_elapsed_time(self) -> float:
        return cp.cuda.get_elapsed_time(self._start_gpu, self._end_gpu) * 1e-3

    def _get_potential(self) -> float:
        r"""Compute local GPU potential and execute a blocking collective :func:`MPI.SUM`
        reduction."""
        local_potential = self.model.compute_potential()
        return cast(float, self.comm.reduce(local_potential, MPI.SUM, root=0))

    def _save_checkpoint(self, ckpt_path: Path) -> None:
        r"""Trigger a collective parallel HDF5 write across all MPI processes."""
        with self.io_manager.open(ckpt_path, "w") as f:
            self.io_manager.write_dict(
                f,
                self.model.get_states(),
                self.model.global_sizes,
                self.model.slices,
            )
            # self.io_manager.write_dict(
            #     f,
            #     self.model.get_estimates(),
            #     self.model.global_sizes,
            #     self.model.slices,
            # )
            self.io_manager.write_stacked(f, "computation_time", self._computation_time)
            self.io_manager.write_rng(f, self.rng)

        with self.io_manager.open_master_only(ckpt_path, "r+") as f:
            if f is not None:
                self.io_manager.write_array(f, "potential", self._potential)

    def _load_checkpoint(self) -> None:
        r"""Load sampler state collaboratively from disk via parallel HDF5 (``mpio``)."""
        with self.io_manager.open(self.restart_path, "r") as f:
            self.model.set_states(
                self.io_manager.read_dict(
                    f,
                    self.model.vars,
                    self.model.slices,
                )
            )
            self.rng = self.io_manager.read_rng(f)
