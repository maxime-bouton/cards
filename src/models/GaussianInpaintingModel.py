from models.BaseModel import BaseModel
from TransitionKernel.TransitionKernel import BaseSerialTransitionKernel, PSGLA

import numpy as np

from operators.jtv import gradient_2d
from functionals.numpy.prox import l21_norm, prox_l21norm

def prox_nonegativity(x):
    return np.maximum(x,0)

def gradient_2d_adjoint(X):

    v = np.zeros_like(X[0,:,:])
    v[0, :] = -X[1,0, :]
    v[1:-1, :] = X[1,:-2, :] - X[1,1:-1, :]  # -np.diff(uv[:-1,:],1,0)
    v[-1, :] = X[1,-2, :]
    v[:, 0] -= X[0,:, 0]
    v[:, 1:-1] += X[0, :, :-2] - X[0, :, 1:-1]  # -np.diff(uv[:,:-1],1,1)
    v[:, -1] += X[0, :, -2]
    return v
#! those two functions must be defined elsewhere


class GaussianInpaintingModel(BaseModel):
    def __init__(self,
                observations : np.ndarray,
                mask : np.ndarray,
                X : BaseSerialTransitionKernel,
                Z : BaseSerialTransitionKernel,
                sigma2 : float,
                reg_coeff : float ,
                split_coeff : float
                ) -> None:
        self.observations = observations
        self.mask = mask
        self.X = X
        self.Z = Z
        self.reg_coeff = reg_coeff
        self.split_coeff = split_coeff
        self.sigma2 = sigma2

        match type(X).__qualname__:
            case PSGLA.__qualname__:
                self.X.prox = prox_nonegativity # implement prox
                self.X.grad = lambda x :  self.mask*( x - self.observations ) / self.sigma2  + gradient_2d_adjoint( self.gradX - self.Z.current_state ) / self.split_coeff            
            case _:
                print("Kernel type not yet supported by this model.") #! move to logger
        
        match type(Z).__qualname__:
            case PSGLA.__qualname__:
                self.Z.prox = lambda z : ( prox_l21norm( z, self.Z.step_size * self.reg_coeff ) )
                self.Z.grad = lambda z : ( z - self.gradX ) / self.split_coeff
            case _:
                print("Kernel type not yet supported by this model.") #! move to logger

        self.gradX = np.zeros( (2, *self.X.current_state.shape) )
        self.MMSE = np.zeros( self.observations.shape ) #! to be moved? -> EstimatorBuilder

    def get_states(self) -> dict:
        states = {}
        states['X'] = self.X.current_state
        states['Z'] = self.Z.current_state
        states['MMSE'] = self.MMSE
        return states
    
    def set_states(self, states: dict) -> None:
        self.X.current_state = states["X"].copy()
        self.Z.current_state = states["Z"].copy()
        self.gradX = gradient_2d(self.X.current_state)
    
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
        return p 