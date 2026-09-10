r"""Serial CPU MCMC sampling backbone."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from pathlib import Path
from time import perf_counter

import numpy as np

from cards.samplers.sampler import Sampler


class SerialCpuSampler(Sampler):
    r"""Serial CPU implementation of the MCMC sampling backbone.

    This class executes the MCMC loop synchronously on a single CPU core, utilizing
    NumPy for random number generation and standard Python high-resolution timers
    (:func:`perf_counter`) for step benchmarking.
    """

    rng: np.random.Generator

    def _setup_rank(self) -> int:
        return 0

    def _start_timer(self) -> None:
        self._step_start = perf_counter()

    def _stop_timer(self) -> None:
        self._step_end = perf_counter()

    def _get_elapsed_time(self) -> float:
        return self._step_end - self._step_start

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
            self.rng = self.io_manager.read_rng(f)  # type : ignore
