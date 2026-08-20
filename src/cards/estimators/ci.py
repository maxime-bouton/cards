r"""Implementation of the MMSE and pixel variance estimates using Welford's online algorithm."""

from collections.abc import Iterable

import cards.backend as xp
from cards.core.execution_context import ExecutionContext
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

    @property
    def declared_keys(self) -> list[str]:
        keys = [f"{self._var.name}_ci"]
        if self._all_samples:
            keys.append(f"{self._var.name}_samples")
        return keys

    @property
    def global_shapes(self) -> dict:
        shapes = super().global_shapes

        if self._all_samples:
            shapes[self.declared_keys[1]] = (self._count, *self._var.layout.full)

        return shapes

    @property
    def slices(self) -> dict:
        slices_dict = super().slices

        if self._all_samples:
            base_slice = self._var.layout.s
            samples_key = self.declared_keys[1]

            if base_slice is not None:
                slices_dict[samples_key] = (slice(None), *base_slice)
            else:
                slices_dict[samples_key] = None

        return slices_dict

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
        self._estimates = {self.declared_keys[0]: quantile_r - quantile_l}
        if self._all_samples:
            self._estimates[self.declared_keys[1]] = valid_samples

    def reset(self) -> None:
        self._count = 0

    def reduce_checkpoints(
        self,
        per_ckpt_estimates: Iterable[dict[str, xp.ndarray]],
        burnin: int,
        ctx: ExecutionContext,
    ) -> dict[str, xp.ndarray]:
        ci_key = self.declared_keys[0]
        if not self._all_samples:
            count = 0
            mean_ci = xp.zeros_like(self._var.state)

            for i, ckpt_dict in enumerate(per_ckpt_estimates):
                if i < burnin:
                    continue
                count += 1

                delta = ckpt_dict[ci_key] - mean_ci
                mean_ci += delta / count

            if count == 0:
                raise ValueError(
                    f"No valid checkpoints after burnin for CI on {self._var.name}"
                )

            return {ci_key: mean_ci}

        else:
            kept_samples = []
            samples_key = self.declared_keys[1]

            for i, ckpt_dict in enumerate(per_ckpt_estimates):
                if i < burnin:
                    continue
                kept_samples.append(ckpt_dict[samples_key])

            if not kept_samples:
                raise ValueError(
                    f"No valid checkpoints after burnin for CI on {self._var.name}"
                )

            all_samples_stacked = xp.concatenate(kept_samples, axis=0)

            quantile_l = xp.quantile(all_samples_stacked, self._alpha / 2, axis=0)
            quantile_r = xp.quantile(all_samples_stacked, 1 - self._alpha / 2, axis=0)

            return {ci_key: quantile_r - quantile_l, samples_key: all_samples_stacked}
