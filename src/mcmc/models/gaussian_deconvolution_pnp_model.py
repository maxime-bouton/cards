"""Implementation of a model used to build a solution to a deconvolution problem under gaussian noise.
Can be executed on cpu or gpu depending on the settings of the backend.py file.
"""

from dataclasses import dataclass

import numpy as np
import torch
from mpi4py import MPI

from mcmc.backend import xp
from mcmc.denoisers.base_denoiser import BaseDenoiser, BaseDistributedDenoiser
from mcmc.models.base_gaussian_deconvolution_model import (
    BaseGaussianDeconvolutionModel,
    GaussianDeconvolutionParams,
)
from mcmc.models.base_model import BaseDistributedModel
from mcmc.operators.dft_convolution import DftConvolution
from mcmc.operators.mpi_dft_convolution import MpiDftConvolution
from mcmc.transition_kernel.base_transition_kernel import (
    BaseTransitionKernel,
)
from mcmc.transition_kernel.gpu_pnp_sgla import GpuPnpSGLA
from mcmc.transition_kernel.gpu_pnp_ula import GpuPnpULA


@dataclass
class GaussianDeconvolutionPnpParams(GaussianDeconvolutionParams): ...


class BaseGaussianDeconvolutionPnpModel(BaseGaussianDeconvolutionModel):
    def __init__(
        self,
        params: GaussianDeconvolutionPnpParams,
        X: BaseTransitionKernel,
        denoiser: BaseDenoiser,
    ):
        self.denoiser = denoiser
        super().__init__(params, X)

    def set_conditionals(self):
        if type(self.X) is GpuPnpULA:
            self.X.denoise = lambda state: self.denoiser(state, self.X.epsilon**0.5)
            self.X.grad = (
                lambda state: self.convolution_operator.adjoint(
                    self.convX - self.observations
                )
                / self.sigma2
            )
            self.X.project = lambda state: state.clip(-1, 2)
        elif type(self.X) is GpuPnpSGLA:
            self.X.denoise = lambda state: self.denoiser(
                state, self.X.reg_coef * self.X.epsilon**0.5
            )
            self.X.grad = (
                lambda state: self.convolution_operator.adjoint(
                    self.convX - self.observations
                )
                / self.sigma2
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

    def get_states(self) -> dict:
        """Extracts the current state of the transition kernel and other variables of interest and return the in a dictionnary.

        Returns
        -------
        dict
            Dictionnary containing the curent states of the variables.
        """
        return {"X": self.X.get_state(), **self._get_estimator_builder_states()}

    def set_states(self, states):
        """Read the dictionnary given in entry and set the variables of the model to the values contained in it.
        The keys used by the dictionnary must be the same as in "get_states"

        Parameters
        ----------
        states : dict
            Dictionnary containing new values for the variables of the model.
        """

        self.X.current_state = xp.asarray(states["X"], dtype=self.X.current_state.dtype)

        self.convX = self.convolution_operator.forward(self.X.current_state)

    def update(self, rng: np.random.Generator | torch.Generator):
        """Global update of the model. Updates every kernel used by the model and computes annex variables.

        Parameters
        ----------
        rng : np.random.Generator | torch.Generator
            Random number generator, given by the sampler.
        """

        self.X.mc_step(rng)

        # update cached buffer related to X
        self.convX = self.convolution_operator.forward(self.X.current_state)

    def compute_potential(self) -> float:
        """Compute the potential of the likelihood for the current step.

        Returns
        -------
        float
            Potential of the targeted distribution.
        """
        p = (0.5 / self.sigma2) * xp.sum((self.observations - self.convX) ** 2)
        return p


class GaussianDeconvolutionPnpModel(BaseGaussianDeconvolutionPnpModel):
    def __init__(
        self,
        params: GaussianDeconvolutionPnpParams,
        X: BaseTransitionKernel,
        denoiser: BaseDenoiser,
    ):
        self.convolution_operator = DftConvolution(
            np.asarray(X.current_state.shape), params.kernel, params.observations.shape
        )

        super().__init__(params, X, denoiser)


class DistributedGaussianDeconvolutionPnpModel(
    BaseGaussianDeconvolutionPnpModel,
    BaseDistributedModel,
):
    def __init__(
        self,
        comm: MPI.Comm,
        full_size: np.ndarray,
        grid_size: np.ndarray,
        params: GaussianDeconvolutionPnpParams,
        X: BaseTransitionKernel,
        denoiser: BaseDistributedDenoiser,
    ):
        self.comm = comm
        self.full_size = full_size

        self.convolution_operator = MpiDftConvolution(
            self.full_size,
            params.kernel,
            self.comm,
            grid_size,
            tile_range=denoiser.tile_range,
        )

        super().__init__(params, X, denoiser)

        self.slices = {}
        self.global_sizes = {}

    def set_slices(self):
        """Describes which portion of the global buffer the current thread must handle."""
        slices = {}
        slices["X"] = self.denoiser.global_to_tile_slice

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

        for key in self.estimator_builder.get_keys():
            var = key.split("_")[0]
            if key.endswith("_samples"):
                sizes[key] = np.r_[self.estimator_builder._batch_size, sizes[var]]
            else:
                sizes[key] = sizes[var]

        self.global_sizes = sizes
