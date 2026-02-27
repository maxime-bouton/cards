"""Implementation of a model used to build a solution to a deconvolution problem under gaussian noise.
Can be executed on cpu or gpu depending on the settings of the backend.py file.
"""

from abc import ABC
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from mcmc.backend import xp
from mcmc.functionals.prox import l21_norm, prox_l21norm, prox_nonegativity
from mcmc.models.base_gaussian_deconvolution_model import (
    BaseGaussianDeconvolutionModel,
    GaussianDeconvolutionParams,
)
from mcmc.models.base_model import BaseDistributedModel
from mcmc.operators.dft_convolution import DftConvolution
from mcmc.operators.gradient import Gradient2d
from mcmc.operators.mpi_dft_convolution import MpiDftConvolution
from mcmc.operators.mpi_gradient import MpiGradient2d
from mcmc.transition_kernel.base_transition_kernel import (
    BaseTransitionKernel,
)
from mcmc.transition_kernel.gpu_psgla import GpuPSGLA
from mcmc.transition_kernel.psgla import PSGLA


@dataclass
class GaussianDeconvolutionTvParams(GaussianDeconvolutionParams):
    split_coeff: float


class BaseGaussianDeconvolutionTvModel(BaseGaussianDeconvolutionModel, ABC):
    gradient_operator: Gradient2d | MpiGradient2d

    def __init__(
        self,
        params: GaussianDeconvolutionTvParams,
        X: BaseTransitionKernel,
        Z: BaseTransitionKernel,
    ):
        self.Z = Z
        self.split_coeff = params.split_coeff
        self.gradX = xp.zeros_like(X.current_state)
        super().__init__(params, X)

    def set_conditionals(self):
        """Set the conditionals of the transition kernels including the coupling between those kernels."""
        if (type(self.X) is PSGLA) or (type(self.X) is GpuPSGLA):
            self.X.prox = prox_nonegativity
            self.X.grad = (
                lambda state: self.convolution_operator.adjoint(
                    self.convX - self.observations
                )
                / self.sigma2
                + self.gradient_operator.adjoint(self.gradX - self.Z.current_state)
                / self.split_coeff
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

        if (type(self.Z) is PSGLA) or (type(self.Z) is GpuPSGLA):
            self.Z.prox = lambda state: (
                prox_l21norm(state, lam=self.Z.step_size * self.reg_coeff)
            )
            self.Z.grad = lambda state: (state - self.gradX) / self.split_coeff
        else:
            raise ValueError("Kernel type not yet supported by this model.")

    def get_states(self) -> dict:
        """Extracts the current state of the transition kernel and other variables of interest and return the in a dictionnary.

        Returns
        -------
        dict
            Dictionnary containing the curent states of the variables.
        """
        return {
            "X": self.X.get_state(),
            "Z": self.Z.get_state(),
            **self._get_estimator_builder_states(),
        }

    def set_states(self, states):
        """Read the dictionnary given in entry and set the variables of the model to the values contained in it.
        The keys used by the dictionnary must be the same as in "get_states"

        Parameters
        ----------
        states : dict
            Dictionnary containing new values for the variables of the model.
        """

        self.X.current_state = xp.asarray(states["X"], dtype=self.X.current_state.dtype)
        self.Z.current_state = xp.asarray(states["Z"], dtype=self.Z.current_state.dtype)

        self.gradX = self.gradient_operator.forward(self.X.current_state)
        self.convX = self.convolution_operator.forward(self.X.current_state)

    def update(self, rng: np.random.Generator):
        """Gobal update of the model. Updates every kernel used by the model and computes annex variables.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator, given by the sampler.
        """

        self.X.mc_step(rng)
        # update cached buffer related to X
        self.gradX = self.gradient_operator.forward(self.X.current_state)
        self.convX = self.convolution_operator.forward(self.X.current_state)

        self.Z.mc_step(rng)

    def compute_potential(self) -> float:
        """compute_potential Compute the potential of the targeted distribution for the current step.

        Returns
        -------
        float
            Potential of the targeted distribution.
        """
        p = (0.5 / self.sigma2) * xp.sum((self.observations - self.convX) ** 2)
        p += xp.sum((self.gradX - self.Z.current_state) ** 2) * (0.5 / self.split_coeff)
        p += self.reg_coeff * l21_norm(self.Z.current_state)
        return p


class GaussianDeconvolutionTvModel(BaseGaussianDeconvolutionTvModel):
    def __init__(
        self,
        params: GaussianDeconvolutionTvParams,
        X: BaseTransitionKernel,
        Z: BaseTransitionKernel,
    ):
        self.gradient_operator = Gradient2d(np.asarray(X.current_state.shape))
        self.convolution_operator = DftConvolution(
            np.asarray(X.current_state.shape), params.kernel, params.observations.shape
        )

        super().__init__(params, X, Z)


class DistributedGaussianDeconvolutionTvModel(
    BaseGaussianDeconvolutionTvModel,
    BaseDistributedModel,
):
    def __init__(
        self,
        comm: MPI.Comm,
        full_size: np.ndarray,
        grid_size: np.ndarray,
        params: GaussianDeconvolutionTvParams,
        X: BaseTransitionKernel,
        Z: BaseTransitionKernel,
    ):
        self.comm = comm
        self.full_size = full_size

        self.gradient_operator = MpiGradient2d(self.full_size, grid_size, self.comm)
        self.convolution_operator = MpiDftConvolution(
            self.full_size,
            params.kernel,
            self.comm,
            grid_size,
        )

        super().__init__(params, X, Z)

        self.slices = {}
        self.global_sizes = {}

    def set_slices(self):
        """Describes which portion of the global buffer the current thread must handle."""
        slices = {}
        slices["X"] = (
            self.gradient_operator.cart_comm.cartslicer._get_slice_global_buffer_to_tile()
        )
        slices["Z"] = (
            np.s_[:],
            *self.gradient_operator.cart_comm.cartslicer._get_slice_global_buffer_to_tile(),
        )

        for key in self.estimator_builder.get_keys():
            var = key.split("_")[0]
            if key.endswith("_samples"):
                slices[key] = (np.s_[:],) + slices[var]
            else:
                slices[key] = slices[var]

        self.slices = slices

    def set_global_sizes(self):
        """Describe the global sizes of several global buffers."""
        sizes = {}
        sizes["X"] = np.asarray(self.full_size, dtype=int)
        sizes["Z"] = np.asarray([2, *self.full_size], dtype=int)

        for key in self.estimator_builder.get_keys():
            var = key.split("_")[0]
            if key.endswith("_samples"):
                sizes[key] = np.r_[self.estimator_builder._batch_size, sizes[var]]
            else:
                sizes[key] = sizes[var]

        self.global_sizes = sizes
