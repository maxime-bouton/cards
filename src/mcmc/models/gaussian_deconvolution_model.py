"""Implementation of a model used to build a solution to a deconvolution problem under gaussian noise.
Can be executed on cpu or gpu depending on the settings of the backend.py file.
"""

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from mcmc.backend import gpu_context, xp
from mcmc.estimator.mmse_builder import mmse_builder, multi_gpu_mmse_builder
from mcmc.functionals.prox import l21_norm, prox_l21norm, prox_nonegativity
from mcmc.models.base_model import BaseModel
from mcmc.operators.dft_convolution import DftConvolution
from mcmc.operators.gradient import Gradient2d
from mcmc.operators.mpi_dft_convolution import MpiDftConvolution
from mcmc.operators.mpi_gradient import MpiGradient2d
from mcmc.transition_kernel.gpu_psgla import GpuPSGLA
from mcmc.transition_kernel.psgla import PSGLA


@dataclass
class DeconvolutionParameters:
    observations: xp.ndarray
    kernel: xp.ndarray
    sigma2: float
    reg_coeff: float
    split_coeff: float


class BaseGaussianDeconvolutionModel(BaseModel):
    def __init__(
        self,
        params: DeconvolutionParameters,
        X,
        Z,
        gpu_id: int = 0,
    ) -> None:
        self.gpu_id = gpu_id

        self.observations = params.observations
        self.convolution_kernel = params.kernel
        self.X = X
        self.Z = Z
        self.reg_coeff = params.reg_coeff
        self.split_coeff = params.split_coeff
        self.sigma2 = params.sigma2

        with gpu_context(self.gpu_id):
            self.gradX = xp.zeros_like(self.X.current_state)
            self.convolution_product = xp.zeros_like(self.observations)

        self.set_conditionals()

    def set_conditionals(self):
        """Set the conditionals of the transition kernels including the coupling between those kernels."""
        if (type(self.X) is PSGLA) or (type(self.X) is GpuPSGLA):
            self.X.prox = prox_nonegativity
            self.X.grad = (
                lambda x: self.convolution_operator.adjoint(
                    self.convolution_product - self.observations
                )
                / self.sigma2
                + self.gradient_operator.adjoint(self.gradX - self.Z.current_state)
                / self.split_coeff
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

        if (type(self.Z) is PSGLA) or (type(self.Z) is GpuPSGLA):
            self.Z.prox = lambda z: (
                prox_l21norm(z, lam=self.Z.step_size * self.reg_coeff)
            )
            self.Z.grad = lambda z: (z - self.gradX) / self.split_coeff
        else:
            raise ValueError("Kernel type not yet supported by this model.")

    def get_states(self) -> dict:
        """get_states
        Extracts the current state of the transition kernel and other variables of interest and return the in a dictionnary.

        Returns
        -------
        dict
            Dictionnary containing the curent states of the variables.
        """
        states = {}
        states["X"] = self.X.get_state()
        states["Z"] = self.Z.get_state()
        if type(self.X) is PSGLA:
            states["MMSE"] = self.estimator_builder.estimator
        if type(self.X) is GpuPSGLA:
            states["MMSE"] = self.estimator_builder.estimator.get()
        return states

    def set_states(self, states):
        """set_states
        Read the dictionnary given in entry and set the variables of the model to the values contained in it.
        The keys used by the dictionnary must be the same as in "get_states"

        Parameters
        ----------
        states : dict
            Dictionnary containing new values for the variables of the model.
        """

        self.X.current_state = states["X"]
        self.Z.current_state = states["Z"]

        self.gradX = self.gradient_operator.forward(self.X.current_state)
        self.convolution_product = self.convolution_operator.forward(
            self.X.current_state
        )

    def aggregate_states(self):
        self.estimator_builder.aggregate_states(self.X.current_state)

    def update(self, rng: np.random.Generator):
        """update Gobal update of the model. Updates every kernel used by the model and computes annex variables.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator, given by the sampler.
        """

        self.X.mc_step(rng)

        # update cached buffer related to X
        self.gradX = self.gradient_operator.forward(self.X.current_state)

        self.convolution_product = self.convolution_operator.forward(
            self.X.current_state
        )

        self.Z.mc_step(rng)

    def compute_potential(self) -> float:
        """compute_potential Compute the potential of the targeted distribution for the current step.

        Returns
        -------
        float
            Potential of the targeted distribution.
        """
        with gpu_context(self.gpu_id):
            p = (0.5 / self.sigma2) * xp.sum(
                (self.observations - self.convolution_product) ** 2
            )
            p += xp.sum((self.gradX - self.Z.current_state) ** 2) * (
                0.5 / self.split_coeff
            )
            p += self.reg_coeff * l21_norm(self.Z.current_state)
            return p


class GaussianDeconvolutionModel(BaseGaussianDeconvolutionModel):
    def __init__(
        self,
        params: DeconvolutionParameters,
        X,
        Z,
        gpu_id=0,
    ):
        self.estimator_builder = mmse_builder(X.current_state.shape)

        self.gradient_operator = Gradient2d(np.asarray(X.current_state.shape))
        self.convolution_operator = DftConvolution(
            np.asarray(X.current_state.shape), params.kernel, params.observations.shape
        )

        super().__init__(
            params,
            X,
            Z,
            gpu_id,
        )


class DistributedGaussianDeconvolutionModel(BaseGaussianDeconvolutionModel):
    def set_slices(self):
        """set_slices Describes which portion of the global buffer the current thread must handle."""
        slices = {}
        slices["X"] = (
            self.gradient_operator.cart_comm.cartslicer._get_slice_global_buffer_to_tile()
        )
        slices["Z"] = (
            np.s_[:],
            *self.gradient_operator.cart_comm.cartslicer._get_slice_global_buffer_to_tile(),
        )
        slices["MMSE"] = (
            self.gradient_operator.cart_comm.cartslicer._get_slice_global_buffer_to_tile()
        )
        self.slices = slices

    def set_global_sizes(self):
        """set_global_sizes Describe the gobla sizes of several global buffers.

        Returns
        -------
        dict
            Global sizes of the variable of interest.
        """
        sizes = {}
        sizes["X"] = np.asarray(self.full_size, dtype=int)
        sizes["Z"] = np.asarray([2, *self.full_size], dtype=int)
        sizes["MMSE"] = np.asarray(self.full_size, dtype=int)

        self.global_sizes = sizes

    def set_local_sizes(self):
        local_sizes = {}
        local_sizes["X"] = self.X.current_state.shape
        local_sizes["Z"] = self.Z.current_state.shape
        local_sizes["MMSE"] = self.estimator_builder.estimator.shape

        self.local_sizes = local_sizes

    def __init__(
        self,
        comm: MPI.Comm,
        full_size: np.ndarray,
        grid_size: np.ndarray,
        params: DeconvolutionParameters,
        X,
        Z,
        gpu_id=0,
    ):
        self.comm = comm
        self.full_size = full_size

        self.gpu_id = gpu_id

        self.estimator_builder = multi_gpu_mmse_builder(
            X.current_state.shape, self.gpu_id
        )
        self.gradient_operator = MpiGradient2d(
            self.full_size, grid_size, self.comm, self.gpu_id
        )
        self.convolution_operator = MpiDftConvolution(
            self.full_size, params.kernel, self.comm, grid_size, self.gpu_id
        )

        super().__init__(
            params,
            X,
            Z,
            gpu_id,
        )

        self.slices = {}
        self.set_slices()

        self.global_sizes = {}
        self.set_global_sizes()

        self.local_sizes = {}
        self.set_local_sizes()

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

        self.gradX = self.gradient_operator.forward(self.X.current_state)
