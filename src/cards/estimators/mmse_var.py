r"""Implementation of the MMSE estimate using an online computing approach."""

import cards.backend as xp
from cards.estimators.base_estimator import BaseEstimator
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel


class MMSEVar(BaseEstimator):
    r"""Implementation of an MMSE estimator.

    Computation is carried out online.
    """

    def __init__(
        self,
        var: BaseTransitionKernel,
        var_name: str = "X",
    ):
        super().__init__(var, var_name)
        self._count = 0
        self._mean = xp.zeros_like(self._var.current_state)
        self._m2 = xp.zeros_like(self._var.current_state)

    def aggregate_states(self) -> None:
        """Update running statistics with new samples using Welford's algorithm."""
        self._count += 1
        delta = self._var.current_state - self._mean
        self._mean += delta / self._count
        delta2 = self._var.current_state - self._mean
        self._m2 += delta * delta2

    def build_estimates(self) -> None:
        """Finalize both mean and variance computations."""
        self._estimates = {
            f"{self._var_name}_mmse": self._mean,
            f"{self._var_name}_var": self._m2 / (self._count - 1),
        }

    def reset(self) -> None:
        self._count = 0
        self._mean.fill(0)
        self._m2.fill(0)
