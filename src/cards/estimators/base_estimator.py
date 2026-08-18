r"""Abstract class used to build Bayesian estimators."""

from abc import ABC, abstractmethod

import cards.backend as xp
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel


class BaseEstimator(ABC):
    """Abstract class underlying the computation of Bayesian estimates.

    Parameters
    ----------
    var : BaseTransitionKernel
        Transition kernel associated with the variable of interest to be estimated.
    var_name : str
        Name of the variable of interest to be estimated.
    """

    def __init__(
        self,
        var: BaseTransitionKernel,
        var_name: str = "X",
        to_cpu: bool = True,
    ) -> None:
        self._var = var
        self._var_name = var_name
        self._estimates: dict[str, xp.ndarray] = {}
        self._to_cpu = to_cpu

    @abstractmethod
    def aggregate_states(self) -> None:
        """Update running statistics with a new sample to compute the estimates with an
        online approach (when possible)."""
        pass

    @abstractmethod
    def build_estimates(self) -> None:
        """Finalize the computation of the estimates."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the value of the estimates to 0."""
        pass

    def get_estimates(self) -> dict[str, xp.ndarray]:
        """Return the final estimates of the variable of interest."""
        if self._to_cpu:
            return {k: v.get() for k, v in self._estimates.items()}
        return self._estimates

    def setup(self, ckpt_size: int) -> None:
        """Setup the estimator to be used for the application of interest.

        Parameters
        ----------
        ckpt_size : int
            Number of samples per checkpoint to be used for the computation of the
            estimates.
        """
        pass
