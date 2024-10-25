"""
Implement a denoising model for the inpainting problem with guassian noise.
"""

from models.BaseModel import BaseModel
from TransitionKernel.TransitionKernel import BaseSerialTransitionKernel, PSGLA
from estimator.estimatorBuilder import MMSEBuilder

import numpy as np

from operators.jtv import gradient_2d
from functionals.numpy.prox import l21_norm, prox_l21norm


def prox_nonegativity(x):
    return np.maximum(x, 0)


def gradient_2d_adjoint(X):
    v = np.zeros_like(X[0, :, :])
    v[0, :] = -X[1, 0, :]
    v[1:-1, :] = X[1, :-2, :] - X[1, 1:-1, :]  # -np.diff(uv[:-1,:],1,0)
    v[-1, :] = X[1, -2, :]
    v[:, 0] -= X[0, :, 0]
    v[:, 1:-1] += X[0, :, :-2] - X[0, :, 1:-1]  # -np.diff(uv[:,:-1],1,1)
    v[:, -1] += X[0, :, -2]
    return v


#! those two functions must be defined elsewhere


class GaussianInpaintingModel(BaseModel):
    def __init__(
        self,
        observations: np.ndarray,
        mask: np.ndarray,
        X: BaseSerialTransitionKernel,
        Z: BaseSerialTransitionKernel,
        sigma2: float,
        reg_coeff: float,
        split_coeff: float,
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

        self.estimator_builder = MMSEBuilder(observations.shape)

        match type(X).__qualname__:
            case PSGLA.__qualname__:
                self.X.prox = prox_nonegativity  # implement prox
                self.X.grad = (
                    lambda x: self.mask * (x - self.observations) / self.sigma2
                    + gradient_2d_adjoint(self.gradX - self.Z.current_state)
                    / self.split_coeff
                )
            case _:
                print("Kernel type not yet supported by this model.")  #! move to logger

        match type(Z).__qualname__:
            case PSGLA.__qualname__:
                self.Z.prox = lambda z: (
                    prox_l21norm(z, self.Z.step_size * self.reg_coeff)
                )
                self.Z.grad = lambda z: (z - self.gradX) / self.split_coeff
            case _:
                print("Kernel type not yet supported by this model.")  #! move to logger

        self.gradX = np.zeros((2, *self.X.current_state.shape))

    def get_states(self) -> dict:
        """get_states
        Exctracts the current state of the transition kernel and other variables of interest and return the in a dictionnary.

        Returns
        -------
        dict
            Dictionnary containing the curent states of the variables.
        """
        states = {}
        states["X"] = self.X.current_state
        states["Z"] = self.Z.current_state
        states["MMSE"] = self.estimator_builder.estimator
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
        self.X.current_state = states["X"].copy()
        self.Z.current_state = states["Z"].copy()
        self.gradX = gradient_2d(self.X.current_state)

    def update(self, rng: np.random.Generator) -> None:
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
        p += np.sum((self.observations - self.mask * self.X.current_state) ** 2) / (
            2 * self.sigma2
        )  # suboptimal
        p += np.sum((self.gradX - self.Z.current_state) ** 2) / (2 * self.split_coeff)
        p += self.reg_coeff * l21_norm(self.Z.current_state)
        return p
