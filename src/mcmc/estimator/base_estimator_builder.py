r"""Abstract class used to build statistic estimators."""

from abc import ABC, abstractmethod

from mcmc.backend import xp


class BaseEstimatorBuilder(ABC):
    @abstractmethod
    def aggregate_states(self, state: xp.ndarray):
        pass

    @abstractmethod
    def build_estimator(self):
        pass

    @abstractmethod
    def reset(self):
        """Reset the internal state of the estimator."""
        pass

    @property
    @abstractmethod
    def estimator(self) -> dict[str, xp.ndarray]:
        """Return the current estimator."""
        pass
