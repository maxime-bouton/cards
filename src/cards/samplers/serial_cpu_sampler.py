r"""Serial CPU MCMC sampling backbone."""

from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from cards.io.io_manager import IOManager
from cards.samplers.base_sampler import BaseSampler


class SerialCpuSampler(BaseSampler):
    r"""Serial CPU implementation of the MCMC sampling backbone.

    This class executes the MCMC loop synchronously on a single CPU core, utilizing
    NumPy for random number generation and standard Python high-resolution timers
    (``perf_counter``) for step benchmarking.
    """

    rng: np.random.Generator

    def _setup_rank(self) -> int:
        return 0

    def _setup_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def _setup_io_manager(self) -> IOManager:
        return IOManager(self.ckpt_size)

    def _start_timer(self) -> None:
        self._step_start = perf_counter()

    def _stop_timer(self) -> None:
        self._step_end = perf_counter()

    def _get_elapsed_time(self) -> float:
        return self._step_end - self._step_start

    def _get_potential(self) -> float:
        return self.model.compute_potential()

    def _save_checkpoint(self, ckpt_path: Path) -> None:
        with h5py.File(ckpt_path, "w") as file:
            self.io_manager.save_dict(self.model.get_states(), file)
            self.io_manager.save_dict(self.model.get_estimates(), file)
            self.io_manager.save_rng(self.rng, file)
            self.io_manager.save_array(self._potential, file, "potential")
            self.io_manager.save_array(self._computation_time, file, "computation_time")

    def _load_checkpoint(self) -> None:
        with h5py.File(self.restart_path, "r") as file:
            self.model.set_states(self.io_manager.load_states(file, self.model.vars))
            self.io_manager.load_rng(self.rng, file)
