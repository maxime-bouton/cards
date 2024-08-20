""" 
    Implement a denoising model for the inpainting problem with guassian noise.
"""
from models.BaseModel import BaseModel
from TransitionKernel.GpuTransitionKernel import BaseGpuTransitionKernel,GpuPSGLA
from estimator.GpuEstimatorBuilder import GpuMMSEBuilder

import numpy as np
import cupy as cp

#from operators.jtv import gradient_2d
#from functionals.numpy.prox import l21_norm, prox_l21norm

def prox_nonegativity(x):
    return cp.maximum(x,0)

def gradient_2d(x):
    uh = cp.zeros_like(x)
    uh[:, :-1] = x[:, 1:] - x[:, :-1]  # np.diff(x,1,1) horizontal differences
    uv = cp.zeros_like(x)
    uv[:-1, :] = x[1:, :] - x[:-1, :]  # np.diff(x,1,0) vertical differences
    return cp.asarray( [uh, uv] )

def gradient_2d_adjoint(X):

    v = cp.zeros_like(X[0,:,:])
    v[0, :] = -X[1,0, :]
    v[1:-1, :] = X[1,:-2, :] - X[1,1:-1, :]  # -np.diff(uv[:-1,:],1,0)
    v[-1, :] = X[1,-2, :]
    v[:, 0] -= X[0,:, 0]
    v[:, 1:-1] += X[0, :, :-2] - X[0, :, 1:-1]  # -np.diff(uv[:,:-1],1,1)
    v[:, -1] += X[0, :, -2]
    return v

def l21_norm(x, axis=0):
    return cp.sum(np.sqrt(np.sum(x**2, axis=axis)))

def prox_l21norm(x, lam=1.0, axis=0):
    if lam <= 0:
        raise ValueError("`lam` should be positive.")
    return x * (1 - 1 / cp.maximum(cp.sqrt(cp.sum(x**2, axis=axis)) / lam, 1.0))

#! those four functions must be defined elsewhere


class GpuGaussianInpaintingModel(BaseModel):
    def __init__(self,
                observations : cp.ndarray,
                mask : cp.ndarray,
                X : BaseGpuTransitionKernel,
                Z : BaseGpuTransitionKernel,
                sigma2 : float,
                reg_coeff : float ,
                split_coeff : float
                ) -> None:
        """
        Parameters
        ----------
        observations : np.ndarray
            Deteriorated picture than can be observed.
        mask : np.ndarray
            Matrix of ones and zeros associated to the inapinting operator.
        X : BaseSerialTransitionKernel
            Random variable following the approximation of the targeted law.
        Z : BaseSerialTransitionKernel
            Splitting variable.
        sigma2 : float
            Standard deviation of the gausssian noise, expexted to be known.
        reg_coeff : float
            Regularisation coefficient.
        split_coeff : float
            Splitting coefficient.
        """
        self.observations = observations
        self.mask = mask
        self.X = X
        self.Z = Z
        self.reg_coeff = reg_coeff
        self.split_coeff = split_coeff
        self.sigma2 = sigma2

        self.estimator_builder = GpuMMSEBuilder( observations.shape )

        match type(X).__qualname__:
            case GpuPSGLA.__qualname__:
                self.X.prox = prox_nonegativity # implement prox
                self.X.grad = lambda x :  self.mask*( x - self.observations ) / self.sigma2  + gradient_2d_adjoint( self.gradX - self.Z.current_state ) / self.split_coeff            
            case _:
                print("Kernel type not yet supported by this model.") #! move to logger
        
        match type(Z).__qualname__:
            case GpuPSGLA.__qualname__:
                self.Z.prox = lambda z : ( prox_l21norm( z, self.Z.step_size * self.reg_coeff ) )
                self.Z.grad = lambda z : ( z - self.gradX ) / self.split_coeff
            case _:
                print("Kernel type not yet supported by this model.") #! move to logger

        self.gradX = cp.zeros( (2, *self.X.current_state.shape) )

    def get_states(self) -> dict:
        """get_states
        Exctracts the current state of the transition kernel and other variables of interest and return the in a dictionnary.

        Returns
        -------
        dict
            Dictionnary containing the curent states of the variables.
        """
        states = {}
        states['X'] = cp.asnumpy( self.X.current_state )
        states['Z'] = cp.asnumpy( self.Z.current_state )
        return states
    
    def set_states(self, states: dict) -> None:
        """set_states 
        Read the dictionnary given in entry and set the variables of the model to the values contained in it.
        The keys used by the dictionnary must be the same as in "get_states"

        Parameters
        ----------
        states : dict
            Dictionnary containing new values for the variables of the model.
        """
        self.X.current_state = cp.asarray( states["X"] ).copy()
        self.Z.current_state = cp.asarray( states["Z"] ).copy()
        self.gradX = gradient_2d(self.X.current_state)
    
    def update(self, rng : cp.random.Generator ) -> None:
        """update Gobal update of the model. Updates every kernel used by the model and computes annex variables.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator, given by the sampler.
        """
        self.X.mc_step(rng)

        self.gradX = gradient_2d(self.X.current_state)

        self.Z.mc_step(rng)

    def aggregate_states(self):
        self.estimator_builder.estimator += self.X.current_state

    def compute_potential(self) -> float:
        """compute_potential Computes the potential.

        Returns
        -------
        float
            Potential of the targeted law.
        """
        p = 0
        p += cp.sum( ( self.observations - self.mask * self.X.current_state)**2 ) / (2 * self.sigma2 ) # suboptimal
        p += cp.sum( (self.gradX - self.Z.current_state) ** 2 ) / (2*self.split_coeff)
        p += self.reg_coeff * l21_norm(self.Z.current_state) #! must be defined on gpu with cupy
        return p 