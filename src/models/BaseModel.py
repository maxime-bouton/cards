from abc import ABC, abstractmethod


class BaseModel(ABC):
    
    @abstractmethod
    def update(self, rng) -> None :
        pass

    @abstractmethod
    def get_states(self) -> dict :
        pass

    @abstractmethod
    def computePotential(self) -> float :
        pass