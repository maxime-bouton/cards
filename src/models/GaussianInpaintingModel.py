from models.BaseModel import BaseModel
from TransitionKernel.TransitionKernel import BaseSerialTransitionKernel, PSGLA

import numpy as np

from operators.jtv import gradient_2d
from functionals.numpy.prox import l21_norm

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
        self.sigma2 = sigma2

        self.gradX = np.zeros( (2, *self.X.current_state.shape) )
        self.MMSE = np.zeros( self.observations.shape ) #! to be moved?

    def get_states(self) -> dict:
        states = {}
        states['X'] = self.X.current_state
        states['Z'] = self.Z.current_state
        states['MMSE'] = self.MMSE
        return states
    
    def update(self, rng) -> None:
        self.X.mc_step(rng)

        self.gradX = gradient_2d(self.X.current_state)

        self.Z.mc_step(rng)

        self.MMSE += self.X.current_state

    def reset_estimator(self) -> None:
        self.MMSE = np.zeros_like(self.X.current_state)
    def normalize_estimator(self, batch_size: int) -> None:
        self.MMSE /= batch_size

    def compute_potential(self) -> float:
        p = 0
        p += np.sum( ( self.observations - self.mask * self.X.current_state)**2 ) / (2 * self.sigma2 ) # suboptimal
        p += np.sum( (self.gradX - self.Z.current_state) ** 2 ) / (2*self.split_coeff)
        p += self.reg_coeff * l21_norm(self.Z.current_state)
        #! to be checked
        return p 