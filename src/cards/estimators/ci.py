r"""Implementation of the MMSE and pixel variance estimates using Welford's online algorithm."""

import cards.backend as xp
from cards.core.variable import Variable
from cards.estimators.base_estimator import BaseEstimator


class CI(BaseEstimator):
    """Batched Credibility Interval estimator using empirical quantiles."""

    def __init__(
        self,
        var: Variable,
        alpha_quantile: float = 0.1,
        all_samples: bool = False,
    ):
        super().__init__(var)
        self._alpha = alpha_quantile
        self._all_samples = all_samples
        self._count = 0

    def setup(self, ckpt_size: int) -> None:
        self._samples = xp.zeros((ckpt_size, *self._var.layout.tile))

    def aggregate_states(self) -> None:
        """Slot the current state into the pre-allocated batch buffer."""
        self._samples[self._count] = self._var.state
        self._count += 1

    def build_estimates(self) -> None:
        """Finalize quantile computations over the valid aggregated samples."""
        valid_samples = self._samples[: self._count]

        quantile_l = xp.quantile(valid_samples, self._alpha / 2, axis=0)
        quantile_r = xp.quantile(valid_samples, 1 - self._alpha / 2, axis=0)
        self._estimates = {f"{self._var.name}_ci": quantile_r - quantile_l}
        if self._all_samples:
            self._estimates[f"{self._var.name}_samples"] = valid_samples

    def reset(self) -> None:
        self._count = 0

    @property
    def global_sizes(self) -> dict:
        sizes = super().global_sizes
        samples_key = f"{self._var.name}_samples"

        if self._all_samples and samples_key in sizes:
            sizes[samples_key] = (self._count, *self._var.layout.full)

        return sizes

    @property
    def slices(self) -> dict:
        slices_dict = super().slices
        samples_key = f"{self._var.name}_samples"

        if self._all_samples and samples_key in slices_dict:
            base_slice = self._var.layout.s

            if base_slice is not None:
                slices_dict[samples_key] = (slice(None), *base_slice)
            else:
                slices_dict[samples_key] = None

        return slices_dict
