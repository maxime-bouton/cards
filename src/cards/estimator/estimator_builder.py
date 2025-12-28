r"""Abstract class used to build Bayesian estimators."""

from abc import ABC, abstractmethod

from cards.backend import xp


class BaseEstimatorBuilder(ABC):
    """Abstract class underlying the computation of Bayesian estimates.

    Attributes
    ----------
    estimator : cards.backend.xp.ndarray
        Internal state of the estimator.
    _name : str
        Name of the variable whose estimate is formed.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: xp.dtype | None = None,
        name: str = "X",
    ):
        self.estimator = xp.zeros(shape, dtype=dtype)
        self._name = name

    @abstractmethod
    def aggregate_states(self, state: xp.ndarray) -> None:
        """Update running statistics with a new sample to compute the estimator with an online approach (when possible)."""
        pass

    @abstractmethod
    def build_estimator(self, N: int) -> None:
        """Finalize the computation of the MMSE estimate."""
        pass

    def reset(self) -> None:
        """Reset the value of the estimator to 0."""
        pass
