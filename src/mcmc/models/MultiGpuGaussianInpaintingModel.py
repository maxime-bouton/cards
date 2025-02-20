r"""Implements a denoising model to solve an inpainting problem under additive white Gaussian noise. Relies on ``cupy`` as a computing backend and ``mpi4py`` for communications."""

import numpy as np
import cupy as cp

from mcmc.distributed_operators.multi_gpu.gradient import distributed_gradient2d

from mcmc.estimator.GPUMMSEBuilder import MultiGpuMMSEBuilder
from mcmc.functionals.gpu.prox import l21_norm, prox_l21norm
from mcmc.models.BaseModel import BaseDistributedModel
from mcmc.TransitionKernel.GpuTransitionKernel import (
    MultiGpuPSGLA,
    BaseGpuTransitionKernel,
)

from mpi4py import MPI


def prox_nonegativity(x):
    return cp.maximum(x, 0)


class MultiGpuGaussianInpaintingModel(BaseDistributedModel):
    def __init__(
        self,
        comm: MPI.Comm,
        full_size: np.ndarray,
        grid_size: np.ndarray,
        observations: cp.ndarray,
        mask: cp.ndarray,
        X: BaseGpuTransitionKernel,
        Z: BaseGpuTransitionKernel,
        sigma2: float,
        reg_coeff: float,
        split_coeff: float,
    ) -> None:
        """
        Parameters
        ----------
        comm: MPI.Comm
            MPI communicator.
        full_size : np.ndarray
            Dimensions of the image.
        grid_size : np.ndarray
            Dimensions of the local subarrays.
        observations : cp.ndarray
            Local subarray of the deteriorated picture than can be observed.
        mask : cp.ndarray
            Local mask. Sub-matrix of ones and zeros associated to the inpainting operator.
        X : BaseSerialTransitionKernel
            Transition kernel for a subarray of the main variable.
        Z : BaseSerialTransitionKernel
            Transition kernel for a subarray of splitting variable.
        sigma2 : float
            Variance of the gaussian noise affecting the observations.
        reg_coeff : float
            Regularisation coefficient.
        split_coeff : float
            Splitting coefficient.
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.full_size = full_size
        self.X = X
        self.Z = Z
        self.reg_coeff = reg_coeff
        self.split_coeff = split_coeff
        self.sigma2 = sigma2

        with cp.cuda.Device(self.rank):
            self.observations = cp.asarray(observations)
            self.mask = cp.asarray(mask)

        self.gradient_handler = distributed_gradient2d(
            np.asarray(self.full_size), grid_size, self.comm
        )
        self.estimator_builder = MultiGpuMMSEBuilder(
            self.gradient_handler.cart_comm.cartslicer.tile_size, self.rank
        )

        with cp.cuda.Device(self.rank):
            self.gradX = np.zeros((2, *self.X.current_state.shape))
            self.adj_buffer = cp.zeros(
                self.gradient_handler.cart_comm.cartslicer.tile_size
            )  #! buffer linked to the kernel used, must be set to 0 due to the implementation of the operator

        self.slices = self.set_slices()
        self.global_sizes = self.set_global_sizes()

        with cp.cuda.Device(self.rank):
            if type(X) is MultiGpuPSGLA:
                self.X.prox = prox_nonegativity
                self.X.grad = (
                    lambda x: self.mask * (x - self.observations) / self.sigma2
                    + self.adj_buffer / self.split_coeff
                )
            else:
                raise ValueError("Kernel type not yet supported by this model.")

            if type(Z) is MultiGpuPSGLA:
                self.Z.prox = lambda z: (
                    prox_l21norm(z, lam=self.Z.step_size * self.reg_coeff)
                )
                self.Z.grad = lambda z: (z - self.gradX) / self.split_coeff
            else:
                raise ValueError("Kernel type not yet supported by this model.")

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

    def set_global_sizes(self) -> dict:
        sizes = {}
        sizes["X"] = np.asarray(self.full_size, dtype=int)
        sizes["Z"] = np.asarray([2, *self.full_size], dtype=int)
        sizes["MMSE"] = np.asarray(self.full_size, dtype=int)

        return sizes

    def get_states(self) -> dict:
        """get_states
        Extracts the current state of the transition kernel and other variables of interest and return it in a dictionnary.

        Returns
        -------
        dict
            Dictionnary containing the curent states of the variables.
        """
        states = {}
        states["X"] = self.X.current_state.get()
        states["Z"] = self.Z.current_state.get()
        states["MMSE"] = self.estimator_builder.estimator.get()
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

        self.gradX = self.gradient_handler.forward(self.X.current_state)

        self.gradient_handler.adjoint(
            # self.adj_buffer,
            self.gradX[0] - self.Z.current_state[0],
            self.gradX[1] - self.Z.current_state[1],
        )

    def aggregate_states(self):
        self.estimator_builder.aggregate_states(self.X.current_state)

    def update(self, rng: np.random.Generator) -> None:
        """update Gobal update of the model. Updates every kernel used by the model and computes annex variables.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator, given by the sampler.
        """

        self.X.mc_step(rng)

        self.gradX = self.gradient_handler.forward(self.X.current_state)

        self.Z.mc_step(rng)

        self.adj_buffer = self.gradient_handler.adjoint(
            self.gradX[0] - self.Z.current_state[0],
            self.gradX[1] - self.Z.current_state[1],
        )

    def compute_potential(self) -> float:
        """compute_potential Computes the partial potential."""
        with cp.cuda.Device(self.rank):
            p = cp.sum((self.observations - self.mask * self.X.current_state) ** 2) / (
                2 * self.sigma2
            )
            p += cp.sum((self.gradX - self.Z.current_state) ** 2) / (
                2 * self.split_coeff
            )
            p += self.reg_coeff * l21_norm(self.Z.current_state)
        return p
