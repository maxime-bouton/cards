r"""Implementation of the MMSE and pixel variance estimates using Welford's online algorithm."""

import cards.backend as xp
from cards.estimators.base_estimator_builder import BaseEstimatorBuilder
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel


class CIBuilder(BaseEstimatorBuilder):
    """Batched Credibility Interval estimator using empirical quantiles."""

    def __init__(
        self,
        var: BaseTransitionKernel,
        var_name: str = "X",
        alpha_quantile: float = 0.1,
        all_samples: bool = False,
    ):
        super().__init__(var, var_name)
        self._alpha = alpha_quantile
        self._all_samples = all_samples
        self._count = 0

    def setup(self, ckpt_size: int) -> None:
        self._samples = xp.zeros((ckpt_size, *self._var.current_state.shape))

    def aggregate_states(self) -> None:
        """Slot the current state into the pre-allocated batch buffer."""
        self._samples[self._count] = self._var.current_state
        self._count += 1

    def build_estimates(self) -> None:
        """Finalize quantile computations over the valid aggregated samples."""
        quantile_l = xp.quantile(self._samples, self._alpha / 2, axis=0)
        quantile_r = xp.quantile(self._samples, 1 - self._alpha / 2, axis=0)
        self._estimates = {f"{self._var_name}_ci": quantile_r - quantile_l}
        if self._all_samples:
            self._estimates.update({f"{self._var_name}_samples": self._samples})

    def reset(self) -> None:
        self._count = 0
