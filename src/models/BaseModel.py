from abc import ABC, abstractmethod
from numpy import random


class BaseModel(ABC):
    
    @abstractmethod
    def update(self, rng : random.Generator) -> None :
        return NotImplemented

    @abstractmethod
    def get_states(self) -> dict :
        pass

    @abstractmethod
    def set_states(self, states : dict) -> None :
        pass

    @abstractmethod
    def compute_potential(self) -> float :
        pass

    @abstractmethod
    def give_data2estimator(self):
        pass