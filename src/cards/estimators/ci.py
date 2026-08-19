r"""Implementation of the MMSE and pixel variance estimates using Welford's online algorithm."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

import cards.backend as xp
from cards.estimators.base_estimator import BaseEstimator
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel


class CI(BaseEstimator):
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
