from src.models.BaseModel import BaseModel
from src.TransitionKernel.TransitionKernel import BaseSerialTransitionKernel, PSGLA

import numpy as np


class GaussianInpaintingModel(BaseModel):
    def __init__(self,
                observations : np.ndarray,
                mask : np.ndarray,
                X : BaseSerialTransitionKernel,
                Z : BaseSerialTransitionKernel,
                sigma2,
                reg_coeff,
                split_coeff
                ) -> None:
        self.observations = observations
        self.mask = mask
        self.X = X
        self.Z = Z
        self.reg_coeff = reg_coeff
        self.split_coeff = split_coeff

        self.MMSE = np.zeros( self.observations.shape ) #! to be moved?

    def get_states(self) -> dict:
        states = {}
        states['X'] = self.X.current_state
        states['Z'] = self.Z.current_state
        states['MMSE'] = self.MMSE
        return states
    
    def update(self, rng) -> None:
        self.X.mc_step(rng)

        #self.gradX = grad2d(X) , to be implemented

        self.Z.mc_step(rng)

    def computePotential(self) -> float:
        p = 0
        #! to be implemented
        return p 