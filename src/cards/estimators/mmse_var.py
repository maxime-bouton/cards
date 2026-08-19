r"""Implementation of the MMSE estimate using an online computing approach."""

import cards.backend as xp
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

        self._estimates = {
            f"{self._var.name}_mmse": self._mean,
            f"{self._var.name}_var": v,
        }

    def reset(self) -> None:
        self._count = 0
        self._mean.fill(0)
        self._m2.fill(0)
