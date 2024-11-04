import numpy as np
from abc import ABC, abstractmethod


class BaseEstimatorBuilder(ABC):
    def __init__(self, shape) -> None:
        self.estimator = np.zeros(shape)

    @abstractmethod
    def aggregate_states(self, state: np.ndarray) -> None:
        pass

    @abstractmethod
    def build_estimator(self, N: int) -> None:
        pass

    def reset(self) -> None:
        self.estimator = np.zeros_like(self.estimator)


class MMSEBuilder(BaseEstimatorBuilder):
    def __init__(self, shape) -> None:
        super().__init__(shape)
        self.name = "MMSE"

    def aggregate_states(self, state: np.ndarray) -> None:
        self.estimator += state

    def build_estimator(self, N: int) -> None:
        self.estimator /= N
