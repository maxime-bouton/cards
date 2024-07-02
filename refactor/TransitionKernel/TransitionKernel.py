import numpy as np

from abc import ABC, abstractmethod

class BaseSerialTransitionKernel(ABC):
    def __init__(self, dims):
        self.currentState = np.zeros( dims )

    @abstractmethod
    def mcStep(self, rng):
        pass

class PSGLA( BaseSerialTransitionKernel ):
    def __init__(self, dims, stepSize):
        super(PSGLA, self).__init__(dims)
        self.stepSize = stepSize

    def prox(self, state : np.ndarray) ->  np.ndarray :
        pass
    def grad(self, state : np.ndarray) ->  np.ndarray :
        pass

    def setProx(self, newProx):
        self.prox = newProx
    
    def setGrad(self, newGrad):
        self.grad = newGrad

    def mcStep(self, rng):
        self.currentState = self.prox(  self.currentState + np.sqrt(2*self.stepSize)*rng.standard_normal( self.currentState.shape ) -self.stepSize*self.grad( self.currentState) )
    #! to be tested