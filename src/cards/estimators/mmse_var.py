r"""Implementation of the MMSE estimate using an online computing approach."""

from collections.abc import Iterable

import cards.backend as xp
from cards.core.execution_context import ExecutionContext
from cards.core.variable import Variable
from cards.estimators.base_estimator import BaseEstimator


class MMSEVar(BaseEstimator):
    r"""Implementation of an MMSE estimator.

    Computation is carried out online.
    """

    def __init__(self, var: Variable):
        super().__init__(var)
        self._count = 0
        self._mean = xp.zeros_like(self._var.state)
        self._m2 = xp.zeros_like(self._var.state)

        self._delta = xp.zeros_like(self._var.state)
        self._delta2 = xp.zeros_like(self._var.state)

    @property
    def declared_keys(self) -> list[str]:
        return [f"{self._var.name}_mmse", f"{self._var.name}_var"]

    def aggregate_states(self) -> None:
        """Update running statistics with new samples using Welford's algorithm."""
        self._count += 1

        xp.subtract(self._var.state, self._mean, out=self._delta)

        self._delta /= self._count
        self._mean += self._delta

        xp.subtract(self._var.state, self._mean, out=self._delta2)

        self._delta *= self._count
        self._delta *= self._delta2
        self._m2 += self._delta

    def build_estimates(self) -> None:
        """Finalize both mean and variance computations."""
        v = self._m2 / (self._count - 1) if self._count > 1 else xp.zeros_like(self._m2)

        mmse_key, var_key = self.declared_keys
        self._estimates = {
            mmse_key: self._mean,
            var_key: v,
        }

    def reset(self) -> None:
        self._count = 0
        self._mean.fill(0)
        self._m2.fill(0)

    def reduce_checkpoints(
        self,
        per_ckpt_estimates: Iterable[dict[str, xp.ndarray]],
        burnin: int,
        ctx: ExecutionContext,
    ) -> dict[str, xp.ndarray]:

        mmse_key, var_key = self.declared_keys

        count = 0
        global_mean = xp.zeros_like(self._var.state)
        m2_of_means = xp.zeros_like(self._var.state)
        sum_of_variances = xp.zeros_like(self._var.state)

        for i, ckpt_dict in enumerate(per_ckpt_estimates):
            if i < burnin:
                continue

            count += 1
            ckpt_mean = ckpt_dict[mmse_key]
            ckpt_var = ckpt_dict[var_key]

            sum_of_variances += ckpt_var

            delta = ckpt_mean - global_mean
            global_mean += delta / count
            delta2 = ckpt_mean - global_mean
            m2_of_means += delta * delta2

        if count == 0:
            raise ValueError("No valid checkpoints after burnin.")

        mean_of_variances = sum_of_variances / count
        variance_of_means = m2_of_means / (count - 1) if count > 1 else 0

        return {mmse_key: global_mean, var_key: mean_of_variances + variance_of_means}
