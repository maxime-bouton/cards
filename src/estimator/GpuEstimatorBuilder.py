import cupy as cp
from abc import ABC, abstractmethod

class BaseGpuEstimatorBuilder(ABC):
    def __init__(self, shape) -> None:
        self.estimator = cp.zeros(shape)

    @abstractmethod
    def aggregate_states(self, state : cp.ndarray) -> None :
        pass

    @abstractmethod
    def build_estimator(self, N : int) -> None :
        pass

    def reset(self) -> None :
        self.estimator = cp.zeros_like(self.estimator)


class GpuMMSEBuilder(BaseGpuEstimatorBuilder):
    def __init__(self, shape) -> None:
        super().__init__(shape)
        self.name = "MMSE"

    def aggregate_states(self, state : cp.ndarray) -> None :
        self.estimator += state
    
    def build_estimator(self, N : int) -> None : 
        self.estimator /= N
