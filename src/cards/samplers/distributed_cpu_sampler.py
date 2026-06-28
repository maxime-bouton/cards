r"""Distributed CPU MCMC sampling backbone."""

import logging
from pathlib import Path
from time import perf_counter
from typing import cast

import h5py
import numpy as np
from mpi4py import MPI

from cards.io.io_manager import IOManager
from cards.models.base_model import BaseDistributedModel
from cards.samplers.base_sampler import BaseSampler, SamplerParameters


class DistributedCpuSampler(BaseSampler):
    r"""MPI-based distributed CPU implementation of the MCMC sampling backbone.

    This class orchestrates parallel MCMC chains across multiple CPU nodes or cores
    using :mod:`mpi4py`. It guarantees statistically independent pseudo-random number
    streams per worker via NumPy :class:`SeedSequence` spawning, and utilizes parallel
    HDF5 (``mpio``) for collective and efficient checkpoint I/O.
    """

    model: BaseDistributedModel
    rng: np.random.Generator

    def __init__(
        self,
        comm: MPI.Comm,
        params: SamplerParameters,
        model: BaseDistributedModel,
        logger: logging.Logger | None = None,
    ):
        self.comm = comm

        super().__init__(params, model, logger)

    def _setup_rank(self) -> int:
        return self.comm.Get_rank()

    def _setup_rng(self) -> np.random.Generator:
        r"""Scatter independent spawned :class:`SeedSequence` streams from root to all
        MPI processes."""
        # set random number generator on each process
        if self.rank == 0:
            ss = np.random.SeedSequence(self.seed)
            # spawn off nworkers child SeedSequences to pass to child processes.
            child_seed = ss.spawn(self.comm.Get_size())
        else:
            child_seed = None
        local_seed = self.comm.scatter(child_seed, root=0)
        return np.random.default_rng(local_seed)

    def _setup_io_manager(self) -> IOManager:
        return IOManager(
            self.ckpt_size,
            self.model.local_sizes,
            self.model.global_sizes,
            self.model.slices,
        )

    def _start_timer(self) -> None:
        self._step_start = perf_counter()

    def _stop_timer(self) -> None:
        self._step_end = perf_counter()

    def _get_elapsed_time(self) -> int | float:
        return self._step_end - self._step_start

    def _get_potential(self) -> float:
        r"""Compute local model potential and execute a blocking collective ``MPI.SUM``
        reduction."""
        local_potential = self.model.compute_potential()
        return cast(float, self.comm.reduce(local_potential, MPI.SUM, root=0))

    def _save_checkpoint(self, ckpt_path: Path) -> None:
        r"""Trigger a collective parallel HDF5 write across all MPI processes."""
        with h5py.File(ckpt_path, "w", driver="mpio", comm=self.comm) as file:
            self.io_manager.save_dict(self.model.get_states(), file)
            self.io_manager.save_dict(self.model.get_estimates(), file)
            self.io_manager.save_rng(self.rng, file, self.rank, self.comm.Get_size())

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
        r"""Load sampler state collaboratively from disk via parallel HDF5 (``mpio``)."""
        with h5py.File(self.restart_path, "r", driver="mpio", comm=self.comm) as file:
            self.model.set_states(self.io_manager.load_states(file, self.model.vars))
            self.io_manager.load_rng(self.rng, file, self.rank)
