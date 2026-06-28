r"""Serial GPU MCMC sampling backbone."""

import logging
from pathlib import Path

import cupy as cp
import h5py
import torch

from cards.io.io_manager import IOManager
from cards.models.base_model import BaseModel
from cards.samplers.base_sampler import BaseSampler, SamplerParameters


class SerialGpuSampler(BaseSampler):
    r"""Serial GPU implementation of the MCMC sampling backbone.

    This class executes the MCMC loop synchronously on a single GPU core, utilizing CuPy
    as backend for GPU computations and PyTorch for random number generation. It employs
    CUDA events for high-resolution timing of each MCMC step.
    """

    rng: torch.Generator

    def __init__(
        self,
        params: SamplerParameters,
        model: BaseModel,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(params, model, logger)

        self._start_gpu = cp.cuda.Event()
        self._end_gpu = cp.cuda.Event()

    def _setup_rank(self) -> int:
        return 0

    def _setup_rng(self) -> torch.Generator:
        return torch.Generator(device="cuda").manual_seed(self.seed)

    def _setup_io_manager(self) -> IOManager:
        return IOManager(self.ckpt_size)

    def _start_timer(self) -> None:
        self._start_gpu.record()

    def _stop_timer(self) -> None:
        r"""Record the GPU stop-event and force a blocking CUDA stream synchronization."""
        self._end_gpu.record()
        self._end_gpu.synchronize()

    def _get_elapsed_time(self) -> float:
        return cp.cuda.get_elapsed_time(self._start_gpu, self._end_gpu) * 1e-3

    def _get_potential(self) -> float:
        return self.model.compute_potential()

    def _save_checkpoint(self, ckpt_path: Path) -> None:
        with h5py.File(ckpt_path, "w") as file:
            self.io_manager.save_dict(self.model.get_states(), file)
            self.io_manager.save_dict(self.model.get_estimates(), file)
            self.io_manager.save_rng_torch(self.rng, self.seed, file)
            self.io_manager.save_array(self._potential, file, "potential")
            self.io_manager.save_array(self._computation_time, file, "computation_time")

    def _load_checkpoint(self) -> None:
        with h5py.File(self.restart_path, "r") as file:
            self.model.set_states(self.io_manager.load_states(file, self.model.vars))
            self.io_manager.load_rng_torch(self.rng, file)
