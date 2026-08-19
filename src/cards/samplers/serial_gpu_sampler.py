r"""Serial GPU MCMC sampling backbone."""

import logging
from pathlib import Path

import cupy as cp
import torch

from cards.estimators.base_estimator import BaseEstimator
from cards.io.io_manager import IOManager
from cards.models.base_model import BaseModel
from cards.samplers.sampler import Sampler, SamplerParameters


class SerialGpuSampler(Sampler):
    r"""Serial GPU implementation of the MCMC sampling backbone.

    This class executes the MCMC loop synchronously on a single GPU core, utilizing
    :mod:`CuPy` as backend for GPU computations and :mod:`torch` for random number
    generation. It employs CUDA events for high-resolution timing of each MCMC step.
    """

    def __init__(
        self,
        io_mng: IOManager,
        params: SamplerParameters,
        model: BaseModel,
        estimators: list[BaseEstimator],
        rng: torch.Generator,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(io_mng, params, model, estimators, rng, logger)

        self._start_gpu = cp.cuda.Event()
        self._end_gpu = cp.cuda.Event()

    def _setup_rank(self) -> int:
        return 0

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
        with self.io_manager.open(ckpt_path, "w") as f:
            self.io_manager.write_dict(f, self.model.states)
            self.io_manager.write_dict(f, self._get_estimates())
            self.io_manager.write_array(f, "potential", self._potential)
            self.io_manager.write_stacked(f, "computation_time", self._computation_time)
            self.io_manager.write_rng(f, self.rng)

    def _load_checkpoint(self) -> None:
        with self.io_manager.open(self.restart_path, "r") as f:
            self.model.states = self.io_manager.read_dict(f, self.model.keys)
            self.rng = self.io_manager.read_rng(f)
