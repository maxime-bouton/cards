r"""Abstract implementation of an MCMC sampler."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import cards.backend as xp
from cards.core.execution_context import ExecutionContext
from cards.io.io_manager import IOManager
from cards.logger import ProgressBar
from cards.models.base_model import BaseModel
from cards.random import create_rng


@dataclass
class SamplerParameters:
    r"""Dataclass gathering all parameters required to run a MCMC sampler
    :class:`~cards.samplers.base_sampler.BaseSampler`."""

    ckpt_size: int
    r"""Number of samples per checkpoint. At the end of each checkpoint, batched
    estimates are computed and saved to disk, along with the current chain state."""

    n_ckpts: int
    r"""Number of checkpoints to compute. The total number of samples is
    ``ckpt_size * n_ckpts``."""

    ckpt_dir_path: Path
    r"""Path to the directory where the checkpoint files are saved."""

    ckpt_prefix: str
    r"""Root name under which checkpoint files will be saved while running the chain."""

    seed: int
    r"""Seed to initialize the random number generator.
    Note that the seed is (deterministically) modified for each MPI worker to ensure
    that different random sequences are generated on each worker."""

    start_ckpt_idx: int = 0
    r"""Index of the checkpoint file to restart from."""

    start_ckpt_dir_path: Path | None = None
    r"""Path to the directory containing the checkpoint file to restart from, by default
    :data:`None` (i.e. same as :attr:`ckpt_dir_path`)."""


class Sampler(ABC):
    r"""Abstract sampler implementation for MCMC algorithms.

    This class handles the main MCMC loop, timing, logging, and I/O operations.

    Parameters
    ----------
    params : SamplerParameters
        Dataclass containing the sampling configuration (e.g., :attr:`ckpt_size`,
        :attr:`n_ckpts`).
    model : BaseModel
        Model encapsulating the MCMC algorithm to be run.
    logger : logging.Logger | None, optional
        Logger object recording the progress of the sampler, by default :data:`None`.

    Attributes
    ----------
    ckpt_size : int
        Number of samples per checkpoint.
    n_ckpts : int
        Number of checkpoints to compute.
    ckpt_dir_path : Path
        Target directory for outputs.
    start_ckpt_idx : int
        Index of the checkpoint file to restart from.
    restart_path : Path | None
        Path to the checkpoint file to restart from.
    model : BaseModel
        The instantiated statistical model.
    io_manager : IOManager
        The I/O handler.
    logger : logging.Logger | None
        The active logger.
    rank : int
        MPI rank of the current worker.
    seed : int
        Base seed for the random number generator.
    rng : np.random.Generator | torch.Generator
        Random number generator instance.

    Methods
    -------
    sample()
        Execute the main MCMC iteration loop.
    """

    def __init__(
        self,
        io_mng: IOManager,
        params: SamplerParameters,
        model: BaseModel,
        rng: np.random.Generator | torch.Generator,
        logger: logging.Logger | None = None,
    ) -> None:
        self.ckpt_size = params.ckpt_size
        self.n_ckpts = params.n_ckpts

        self._ckpt_file = f"{params.ckpt_prefix}{{}}.h5"
        self.ckpt_dir_path = params.ckpt_dir_path

        self.start_ckpt_idx = params.start_ckpt_idx
        start_path = params.start_ckpt_dir_path or self.ckpt_dir_path
        self.restart_path = start_path / self._ckpt_file.format(self.start_ckpt_idx)

        self.model = model
        self.io_manager = io_mng
        self.logger = logger

        self.rank = self._setup_rank()
        self.rng = rng

        self.seed = params.seed

        if self.rank == 0:
            self._potential = xp.zeros([self.ckpt_size])

        self._computation_time = xp.zeros([self.ckpt_size])

    @classmethod
    def create_from_context(
        cls,
        ctx: ExecutionContext,
        io_mng: IOManager,
        model: BaseModel,
        params: SamplerParameters,
        logger: logging.Logger | None = None,
    ) -> "Sampler":
        r"""Factory method to instantiate the correct child class."""

        from . import (
            DistributedCpuSampler,
            DistributedGpuSampler,
            SerialCpuSampler,
            SerialGpuSampler,
        )

        rng = create_rng(params.seed, ctx)

        match (ctx.is_mpi, ctx.is_gpu):
            case (False, False):
                return SerialCpuSampler(io_mng, params, model, rng, logger)
            case (True, False):
                return DistributedCpuSampler(
                    io_mng,
                    ctx.comm,
                    params,
                    model,  # type: ignore
                    rng,  # type: ignore
                    logger,
                )
            case (False, True):
                return SerialGpuSampler(io_mng, params, model, rng, logger)  # type: ignore
            case (True, True):
                return DistributedGpuSampler(
                    io_mng,
                    ctx.comm,
                    params,
                    model,  # type: ignore
                    rng,  # type: ignore
                    logger,
                )
            case _:
                raise ValueError(
                    f"Unknown configuration: mpi={ctx.is_mpi}, gpu={ctx.is_gpu}"
                )

    @abstractmethod
    def _setup_rank(self) -> int:
        r"""Determine and return the local worker rank."""

    @abstractmethod
    def _start_timer(self) -> None:
        r"""Start the timer for the current MCMC step."""

    @abstractmethod
    def _stop_timer(self) -> None:
        r"""Stop the timer for the current MCMC step."""

    @abstractmethod
    def _get_elapsed_time(self) -> float:
        r"""Return the measured elapsed time of the last step in seconds."""

    @abstractmethod
    def _get_potential(self) -> float:
        r"""Return the current (local) potential of the model."""

    @abstractmethod
    def _save_checkpoint(self, ckpt_path: Path) -> None:
        r"""Trigger the IOManager to write the current checkpoint to disk."""

    @abstractmethod
    def _load_checkpoint(self) -> None:
        r"""Load the sampler state from a previous checkpoint."""

    def sample(self) -> None:
        r"""Main iteration loop of the MCMC algorithm.

        At each iteration, calls the update step of the model. The current state
        of the parameters is regularly saved in checkpoint files, along with
        quantities required for a batched evaluation of the final estimates.
        """
        if self.start_ckpt_idx > 0:
            self._load_checkpoint()

        if self.rank == 0:
            pbar = ProgressBar(total=self.n_ckpts, desc="Sampling", bar_len=100)
            pbar.update(self.start_ckpt_idx)

        self.model.setup_estimators(self.ckpt_size)

        for ckpt_idx in range(self.start_ckpt_idx, self.n_ckpts):
            self.model.reset_estimators()

            for i in range(self.ckpt_size):
                self._start_timer()
                self.model.update(self.rng)
                self._stop_timer()

                potential = self._get_potential()
                if self.rank == 0:
                    self._potential[i] = potential

                self.model.aggregate_states()
                self._computation_time[i] = self._get_elapsed_time()

            self.model.build_estimates()

            ckpt_path = self.ckpt_dir_path / self._ckpt_file.format(ckpt_idx + 1)
            self._save_checkpoint(ckpt_path)

            if self.rank == 0:
                pbar.clear()
                if self.logger:
                    pad = len(str(self.n_ckpts))
                    cyan = "\033[96m"
                    yellow = "\033[93m"
                    magenta = "\033[95m"
                    reset = "\033[0m"
                    self.logger.info(
                        f"  │    │ Checkpoint {cyan}{ckpt_idx + 1:>{pad}}/{self.n_ckpts}{reset} | "
                        f"Potential: {yellow}{self._potential[-1]: 1.3e}{reset} | "
                        f"t/it (s): {magenta}{self._computation_time[-1]: 1.3e}{reset}"
                    )
                ckpt_time = sum(self._computation_time)
                pbar.update(ckpt_idx + 1, time_per_step=ckpt_time)

        if self.rank == 0:
            pbar.clear()
