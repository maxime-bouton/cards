import numpy as np

class mmse_builder():
    def __init__(self, shape) -> None:
        self.estimator = np.zeros(shape)

    def update(self, state : np.ndarray) -> None :
        self.estimator += state
    
    def normalize(self, N : int) -> None : 
        self.estimator /= N
    
    def reset(self) -> None :
        self.estimator = np.zeros_like(self.estimator)
