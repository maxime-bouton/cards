r"""Abstract class used to build Bayesian estimators."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

import cards.backend as xp
from cards.core.execution_context import ExecutionContext
from cards.core.variable import Variable


class BaseEstimator(ABC):
    r"""Abstract class underlying the computation of Bayesian estimates.

    Parameters
    ----------
    var : Variable
        Variable of interest to be estimated.
    """

    def __init__(self, var: Variable) -> None:
        self._var = var
        self._estimates = {}

    @property
    def estimates(self) -> dict[str, xp.ndarray]:
        r"""Return the estimates of the variable of interest."""
        return self._estimates

    @property
    def local_shapes(self) -> dict:
        r"""Return the shapes of the local buffers representing the portion of the
        estimates handled by each MPI worker.
        """
        return {n: self._var.layout.tile for n in self.declared_keys}

    @property
    def global_shapes(self) -> dict:
        r"""Return the shapes of the global buffers representing the whole estimates."""
        return {n: self._var.layout.full for n in self.declared_keys}

    @property
    def slices(self) -> dict:
        r"""Return the slices to select the local portion of each estimate from their
        corresponding global buffers."""
        return {n: self._var.layout.s for n in self.declared_keys}

    @property
    @abstractmethod
    def declared_keys(self) -> list[str]:
        r"""List of keys that are used by the estimator. Entirely known from `var` alone,
        no :class:`~cards.models.base_model.BaseModel` needed. Single source of truth
        for the estimate keys contained in checkpoint files.
        """

    def setup(self, ckpt_size: int) -> None:
        r"""Setup the estimator to be used for the application of interest.

        Parameters
        ----------
        ckpt_size : int
            Number of samples per checkpoint to be used for the computation of the
            estimates.
        """

    @abstractmethod
    def aggregate_states(self) -> None:
        r"""Update running statistics with a new sample to compute the estimates with an
        online approach (when possible)."""

    @abstractmethod
    def build_estimates(self) -> None:
        r"""Finalize the computation of the estimates."""

    @abstractmethod
    def reset(self) -> None:
        r"""Reset the value of the estimates to 0."""

    @abstractmethod
    def reduce_checkpoints(
        self,
        per_ckpt_estimates: Iterable[dict[str, xp.ndarray]],
        burnin: int,
        ctx: ExecutionContext,
    ) -> dict[str, xp.ndarray]:
        r"""Combine per-checkpoint batched estimates. An estimator whose math genuinely
        needs neighboring data should reach for `ctx.comm` itself."""
