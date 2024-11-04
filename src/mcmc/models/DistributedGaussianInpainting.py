"""
Implement a denoising model for the inpainting problem with guassian noise.
"""

from mcmc.models.BaseModel import BaseDistributedModel
from mcmc.TransitionKernel.TransitionKernel import BaseSerialTransitionKernel, PSGLA
from mcmc.estimator.estimatorBuilder import MMSEBuilder
from mcmc.distributed_operators.gradient import distributed_gradient2d

import numpy as np

from mcmc.functionals.numpy.prox import l21_norm, prox_l21norm


def prox_nonegativity(x):
    return np.maximum(x, 0)


class DistributedGaussianInpaintingModel(BaseDistributedModel):
    def __init__(
        self,
        full_size: np.ndarray,
        grid_size: np.ndarray,
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
        full_size : np.ndarray
            Dimensions of the image.
        grid_size : np.ndarray
            Dimensions of the local subarrays.
        observations : np.ndarray
            Local subarray of the deteriorated picture than can be observed.
        mask : np.ndarray
            Local mask. Sub-matrix of ones and zeros associated to the inapinting operator.
        X : BaseSerialTransitionKernel
            Subarray of the random variable following the approximation of the targeted law.
        Z : BaseSerialTransitionKernel
            Subarray of the plitting variable.
        sigma2 : float
            Standard deviation of the gausssian noise, expexted to be known.
        reg_coeff : float
            Regularisation coefficient.
        split_coeff : float
            Splitting coefficient.
        """
        self.full_size = full_size
        self.observations = observations
        self.mask = mask
        self.X = X
        self.Z = Z
        self.reg_coeff = reg_coeff
        self.split_coeff = split_coeff
        self.sigma2 = sigma2

        self.gradient_handler = distributed_gradient2d(
            np.asarray(self.full_size), grid_size
        )
        self.adj_buffer = np.zeros(
            self.gradient_handler.cart_comm.cartslicer.tile_size
        )  #! buffer linked to the kernel used

        self.slices = self.set_slices()
        self.global_sizes = self.set_global_sizes()

        self.estimator_builder = MMSEBuilder(
            self.gradient_handler.cart_comm.cartslicer.tile_size
        )

        match type(X).__qualname__:
            case PSGLA.__qualname__:
                self.X.prox = prox_nonegativity  # implement prox
                self.X.grad = (
                    lambda x: self.mask * (x - self.observations) / self.sigma2
                    + self.adj_buffer / self.split_coeff
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

    def set_slices(self) -> dict:
        slices = {}
        slices["X"] = (
            self.gradient_handler.cart_comm.cartslicer._get_slice_global_buffer_to_tile()
        )
        slices["Z"] = (
            np.s_[:],
            *self.gradient_handler.cart_comm.cartslicer._get_slice_global_buffer_to_tile(),
        )
        slices["MMSE"] = (
            self.gradient_handler.cart_comm.cartslicer._get_slice_global_buffer_to_tile()
        )
        return slices

    #! local buffer
    def get_states(self) -> dict:
        """get_states
        Exctracts the current state of the transition kernel and other variables of interest and return it in a dictionnary.

        Returns
        -------
        dict
            Dictionnary containing the curent states of the variables.
        """
        states = {}
        states["X"] = self.X.current_state
        states["Z"] = self.Z.current_state  #! pb on slices
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

        self.gradX = self.gradient_handler.compute_grad(self.X.current_state)
        self.gradient_handler.compute_adjoint(
            self.adj_buffer,
            self.gradX[0] - self.Z.current_state[0],
            self.gradX[1] - self.Z.current_state[1],
        )

    def set_global_sizes(self) -> dict:
        sizes = {}
        sizes["X"] = np.asarray(self.full_size, dtype=int)
        sizes["Z"] = np.asarray([2, *self.full_size], dtype=int)
        sizes["MMSE"] = np.asarray(self.full_size, dtype=int)

        return sizes

    def update(self, rng: np.random.Generator) -> None:
        """update Gobal update of the model. Updates every kernel used by the model and computes annex variables.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator, given by the sampler.
        """

        self.X.mc_step(rng)

        self.gradX = self.gradient_handler.compute_grad(self.X.current_state)

        self.Z.mc_step(rng)

        self.adj_buffer[:] = 0
        self.gradient_handler.compute_adjoint(
            self.adj_buffer,
            self.gradX[0] - self.Z.current_state[0],
            self.gradX[1] - self.Z.current_state[1],
        )

    def aggregate_states(self):
        self.estimator_builder.estimator += self.X.current_state

    def compute_potential(self) -> float:
        """compute_potential Computes the partial potential."""
        p = 0
        p += np.sum((self.observations - self.mask * self.X.current_state) ** 2) / (
            2 * self.sigma2
        )  # suboptimal
        p += np.sum((self.gradX - self.Z.current_state) ** 2) / (2 * self.split_coeff)
        p += self.reg_coeff * l21_norm(self.Z.current_state)
        return p
