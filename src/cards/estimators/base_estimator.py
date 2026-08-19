r"""Abstract class used to build Bayesian estimators."""

from abc import ABC, abstractmethod

import cards.backend as xp
from cards.core.variable import Variable


class BaseEstimator(ABC):
    """Abstract class underlying the computation of Bayesian estimates.

    Parameters
    ----------
    var : Variable
        Variable of interest to be estimated.
    """

    def __init__(self, var: Variable) -> None:
        self._var = var
        self._estimates: dict[str, xp.ndarray] = {}

    def setup(self, ckpt_size: int) -> None:
        """Setup the estimator to be used for the application of interest.

        Parameters
        ----------
        ckpt_size : int
            Number of samples per checkpoint to be used for the computation of the
            estimates.
        """

    @abstractmethod
    def aggregate_states(self) -> None:
        """Update running statistics with a new sample to compute the estimates with an
        online approach (when possible)."""

    @abstractmethod
    def build_estimates(self) -> None:
        """Finalize the computation of the estimates."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the value of the estimates to 0."""

    @property
    def estimates(self) -> dict[str, xp.ndarray]:
        """Return the final estimates of the variable of interest."""
        return self._estimates

    @property
    def local_sizes(self) -> dict:
        r"""Return sizes of the local buffers representing the portion of the variables
        handled by each MPI worker."""
        return {n: self._var.layout.tile for n in self._estimates}

    @property
    def global_sizes(self) -> dict:
        r"""Return the sizes of the global buffers representing the variables to be sampled."""
        return {n: self._var.layout.full for n in self._estimates}

    @property
    def slices(self) -> dict:
        r"""Return the slices to select local variables from the global memory buffer."""
        return {n: self._var.layout.s for n in self._estimates}
