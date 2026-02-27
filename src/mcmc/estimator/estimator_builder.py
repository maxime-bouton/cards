from typing import TypeAlias

from mcmc.backend import xp
from mcmc.estimator.base_estimator_builder import BaseEstimatorBuilder

ShapeLike: TypeAlias = tuple[int, ...]


class EstimatorBuilder(BaseEstimatorBuilder):
    def __init__(self, dict_variables: dict[str, tuple[ShapeLike, xp.dtype | None]]):
        super().__init__()
        self._dict_var = dict_variables

        self._count = 0
        self._means = {k: xp.zeros(v[0], dtype=v[1]) for k, v in self._dict_var.items()}
        self._m2s = {k: xp.zeros(v[0], dtype=v[1]) for k, v in self._dict_var.items()}

        self._batch_size = None
        self._compute_ci = False
        self._save_all = False

    def set_batch_size(self, batch_size: int):
        self._batch_size = batch_size
        self._samples = {
            k: xp.zeros((batch_size, *v[0]), dtype=v[1])
            for k, v in self._dict_var.items()
        }

    def set_alpha_quantile(self, alpha_q: float):
        if not self._compute_ci:
            raise ValueError(
                "Confidence intervals must be enabled before setting alpha quantile."
            )
        if not (0 < alpha_q < 1):
            raise ValueError("Alpha quantile must be in the range (0, 1).")
        self._alpha_q = alpha_q

    def enable_compute_ci(self):
        if self._batch_size is None:
            raise ValueError(
                "Batch size must be set before computing confidence intervals."
            )
        self._alpha_q = 0.05
        self._compute_ci = True

    def enable_save_all(self):
        """Enable saving all samples for post-process analysis."""
        if self._batch_size is None:
            raise ValueError("Batch size must be set before enabling save_all.")
        self._save_all = True

    def aggregate_states(self, state: dict[str, xp.ndarray]):
        """Update running statistics with new samples using Welford's algorithm.

        Parameters
        ----------
        state : dict[str, xp.ndarray]
            New samples to incorporate into estimates.
        """
        for k, v in state.items():
            if k not in self._means:
                raise ValueError(f"Variable {k} not recognized.")
            if v.shape != self._dict_var[k][0]:
                raise ValueError(
                    f"Shape mismatch for variable {k}: "
                    f"expected {self._dict_var[k][0]}, got {v.shape}."
                )
            self._count += 1
            delta = v - self._means[k]
            self._means[k] += delta / self._count
            delta2 = v - self._means[k]
            self._m2s[k] += delta * delta2

            if self._batch_size is not None:
                self._samples[k][self._count - 1] = v

    def build_estimator(self):
        """Build the final estimator with computed statistics."""

        self._estimator = {}
        for k in self._means:
            self._estimator[f"{k}_mmse"] = self._means[k]
            self._estimator[f"{k}_var"] = self._m2s[k] / (self._count - 1)

            if self._compute_ci:
                quantile_l = xp.quantile(self._samples[k], self._alpha_q / 2, axis=0)
                quantile_r = xp.quantile(
                    self._samples[k], 1 - self._alpha_q / 2, axis=0
                )
                self._estimator[f"{k}_CI"] = quantile_r - quantile_l

            if self._save_all:
                self._estimator[f"{k}_samples"] = self._samples[k]

    def reset(self):
        self._count = 0
        for k in self._means:
            self._means[k].fill(0)
            self._m2s[k].fill(0)
            if self._batch_size is not None:
                self._samples[k].fill(0)

    def get_keys(self) -> list[str]:
        """Return the keys of the estimator."""
        keys = []
        for k in self._means:
            keys.append(f"{k}_mmse")
            keys.append(f"{k}_var")
            if self._compute_ci:
                keys.append(f"{k}_CI")
            if self._save_all:
                keys.append(f"{k}_samples")
        return keys

    @property
    def estimator(self) -> dict[str, xp.ndarray]:
        return self._estimator
