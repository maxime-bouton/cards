r"""Implementation of a Gaussian deconvolution model under a PnP prior to reproduce the experiments reported in :cite:p:`Bouton2025`."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.

# TODO: documentation

from dataclasses import dataclass

import numpy as np
import torch

from cards.backend import xp
from cards.denoisers.base_denoiser import BaseDenoiser, BaseDistributedDenoiser
from cards.models.base_gaussian_deconvolution_model import (
    BaseGaussianDeconvolutionModel,
    GaussianDeconvolutionParams,
)
from cards.models.base_model import BaseDistributedModel
from cards.operators.dft_convolution import DftConvolution
from cards.operators.mpi_dft_convolution import MpiDftConvolution
from cards.transition_kernel.base_transition_kernel import (
    BaseTransitionKernel,
)
from cards.transition_kernel.gpu_pnp_sgla import GpuPnpSGLA
from cards.transition_kernel.gpu_pnp_ula import GpuPnpULA


@dataclass
class GaussianDeconvolutionPnpParams(GaussianDeconvolutionParams): ...


class BaseGaussianDeconvolutionPnpModel(BaseGaussianDeconvolutionModel):
    def __init__(
        self,
        params: GaussianDeconvolutionPnpParams,
        convolution_operator: DftConvolution | MpiDftConvolution,
        X: BaseTransitionKernel,
        denoiser: BaseDenoiser,
    ):
        self.denoiser = denoiser
        super().__init__(params, convolution_operator, X)

    def set_conditionals(self):
        if type(self.X) is GpuPnpULA:
            self.X.denoise = lambda state: self.denoiser(
                state,
                self.X.epsilon**0.5,
                torch_dtype=torch.float32,
                cp_dtype=xp.float64,
            )
            self.X.grad = lambda state: (
                self.convolution_operator.adjoint(self.convX - self.observations)
                / self.sigma2
            )
            self.X.project = lambda state: state.clip(-1, 2)
        elif type(self.X) is GpuPnpSGLA:
            self.X.denoise = lambda state: self.denoiser(
                state,
                self.X.reg_coef * self.X.epsilon**0.5,
                torch_dtype=torch.float32,
                cp_dtype=xp.float64,
            )
            self.X.grad = lambda state: (
                self.convolution_operator.adjoint(self.convX - self.observations)
                / self.sigma2
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

    def get_states(self) -> dict:
        """Extracts the current state of the transition kernel and other variables of interest and return the in a dictionary.

        Returns
        -------
        dict
            Dictionary containing the curent states of the variables.
        """
        return {"X": self.X.get_state(), "MMSE": self.estimator_builder.estimator.get()}

    def set_states(self, states):
        """Read the dictionary given in entry and set the variables of the model to the values contained in it.
        The keys used by the dictionary must be the same as in "get_states"

        Parameters
        ----------
        states : dict
            Dictionary containing new values for the variables of the model.
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
        convolution_operator: DftConvolution,
        params: GaussianDeconvolutionPnpParams,
        X: BaseTransitionKernel,
        denoiser: BaseDenoiser,
    ):
        # self.convolution_operator = DftConvolution(
        #     np.asarray(X.current_state.shape), params.kernel, params.observations.shape
        # )

        super().__init__(params, convolution_operator, X, denoiser)


class DistributedGaussianDeconvolutionPnpModel(
    BaseGaussianDeconvolutionPnpModel,
    BaseDistributedModel,
):
    def __init__(
        self,
        convolution_operator: MpiDftConvolution,
        params: GaussianDeconvolutionPnpParams,
        X: BaseTransitionKernel,
        denoiser: BaseDistributedDenoiser,
    ):
        self.full_size = convolution_operator.image_size

        # self.convolution_operator = MpiDftConvolution(
        #     self.full_size,
        #     params.kernel,
        #     self.comm,
        #     grid_size,
        # )
        super().__init__(params, convolution_operator, X, denoiser)

    def set_slices(self):
        """Describes which portion of the global buffer the current thread must handle."""
        self.slices["X"] = self.denoiser.global_to_tile_slice
        self.slices["MMSE"] = self.denoiser.global_to_tile_slice

    def set_global_sizes(self):
        """Describe the global sizes of several global buffers."""
        self.global_sizes["X"] = np.asarray(self.full_size, dtype=int)
        self.global_sizes["MMSE"] = np.asarray(self.full_size, dtype=int)

    def set_local_sizes(self):
        self.local_sizes["X"] = self.X.current_state.shape
        self.local_sizes["MMSE"] = self.X.current_state.shape
