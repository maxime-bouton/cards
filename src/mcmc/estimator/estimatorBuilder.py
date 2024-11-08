r"""Abstract class used to build statistic estimators."""

from abc import ABC, abstractmethod
from typing import Any


class BaseEstimatorBuilder(ABC):
    estimator: Any
    """Internal state of the estimator, its internal_state depends on the implementation"""

    __name: str
    """A human readable name for this estimator"""

    @abstractmethod
    def aggregate_states(self, state: Any) -> None:
        pass

    @abstractmethod
    def build_estimator(self, N: int) -> None:
        pass

    def reset(self) -> None:
        """Reset the internal state of the estimator."""
        pass


class BaseMMSEBuilder(BaseEstimatorBuilder):
    r"""Numpy implementation of a MMSE estimator.

    Numpy implementation of :class:`mcmc.estimator.estimatorBuilder.BaseMMSEBuilder`.

    Attributes
    ----------
    estimator : Any
        Internal state of the MMSE estimator.
    __name : str
        Name of the estimator.
    """

    def __init__(self):
        self.__name = "MMSE"

    def aggregate_states(self, state: Any) -> None:
        self.estimator += state

    def build_estimator(self, N: int) -> None:
        self.estimator /= N
