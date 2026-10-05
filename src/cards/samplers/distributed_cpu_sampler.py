r"""Distributed CPU MCMC sampling backbone."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

import logging
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
from mpi4py import MPI

from cards.estimators.base_estimator import BaseEstimator
from cards.io.io_manager import IOManager
from cards.models.base_model import BaseDistributedModel
from cards.samplers.sampler import Sampler, SamplerParameters


class DistributedCpuSampler(Sampler):
    r"""MPI-based distributed CPU implementation of the MCMC sampling backbone.

    This class orchestrates parallel MCMC chains across multiple CPU nodes or cores
    using :mod:`mpi4py`. It guarantees statistically independent pseudo-random number
    streams per worker via NumPy :class:`SeedSequence` spawning, and utilizes parallel
    HDF5 (``mpio``) for collective and efficient checkpoint I/O.
    """

    model: BaseDistributedModel

    def __init__(
        self,
        io_mng: IOManager,
        comm: MPI.Comm,
        params: SamplerParameters,
        model: BaseDistributedModel,
        estimators: list[BaseEstimator],
        rng: np.random.Generator,
        logger: logging.Logger | None = None,
    ):
        self.comm = comm

        super().__init__(io_mng, params, model, estimators, rng, logger)

    def _setup_rank(self) -> int:
        return self.comm.Get_rank()

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
        with self.io_manager.open(ckpt_path, "w") as f:
            self.io_manager.write_dict(
                f,
                self.model.states,
                self.model.global_sizes,
                self.model.slices,
            )
            self.io_manager.write_dict(
                f,
                self._get_estimates(),
                self._get_estimates_global_sizes(),
                self._get_estimates_slices(),
            )
            self.io_manager.write_stacked(f, "computation_time", self._computation_time)
            self.io_manager.write_rng(f, self.rng)

        with self.io_manager.open_master_only(ckpt_path, "r+") as f:
            if f is not None:
                self.io_manager.write_array(f, "potential", self._potential)

    def _load_checkpoint(self) -> None:
        r"""Load sampler state collaboratively from disk via parallel HDF5 (``mpio``)."""
        with self.io_manager.open(self.restart_path, "r") as f:
            self.model.states = self.io_manager.read_dict(
                f,
                self.model.keys,
                self.model.slices,
            )
            self.rng = self.io_manager.read_rng(f)

    def _get_estimates_global_sizes(self):
        estimates_global_sizes = {}
        for estimator in self.estimators:
            estimates_global_sizes.update(estimator.global_shapes)
        return estimates_global_sizes

    def _get_estimates_slices(self):
        estimates_slices = {}
        for estimator in self.estimators:
            estimates_slices.update(estimator.slices)
        return estimates_slices
