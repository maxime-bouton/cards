from abc import ABC, abstractmethod


class BaseModel(ABC):
    
    @abstractmethod
    def update(self, rng) -> None :
        pass

    @abstractmethod
    def get_states(self) -> dict :
        pass

    @abstractmethod
    def compute_potential(self) -> float :
        pass

    @abstractmethod
    def reset_estimator(self) -> None:
        pass

    @abstractmethod
    def normalize_estimator(self, batch_size : int ) -> None :
        pass
