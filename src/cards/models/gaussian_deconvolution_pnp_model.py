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

import cards.backend as xp
from cards.core.variable import Variable
from cards.denoisers.base_denoiser import BaseDenoiser
from cards.models.base_gaussian_deconvolution_model import (
    BaseGaussianDeconvolutionModel,
    GaussianDeconvolutionParams,
)
from cards.models.base_model import BaseDistributedModel
from cards.operators.dft_convolution import DftConvolution
from cards.operators.mpi_dft_convolution import MpiDftConvolution
from cards.transition_kernels.base_transition_kernel import (
    BaseTransitionKernel,
)
from cards.transition_kernels.gpu_pnp_sgla import GpuPnpSGLA
from cards.transition_kernels.pnp_ula import PnpULA


@dataclass
class GaussianDeconvolutionPnpParams(GaussianDeconvolutionParams): ...


class GaussianDeconvolutionPnpModel(BaseGaussianDeconvolutionModel):
    def __init__(
        self,
        params: GaussianDeconvolutionPnpParams,
        convolution_operator: DftConvolution | MpiDftConvolution,
        y: Variable,
        X: BaseTransitionKernel,
        denoiser: BaseDenoiser,
    ):
        super().__init__(params, convolution_operator, y, X)
        self.denoiser = denoiser

    def set_conditionals(self):
        if isinstance(self.X, PnpULA):
            self.X.denoise = lambda state: self.denoiser(state, self.X.epsilon**0.5)
            self.X.grad = lambda state: (
                self.H.adjoint(self.Hx - self.y.state) / self.sigma2
            )
            self.X.project = lambda state: state.clip(-1, 2)
        elif isinstance(self.X, GpuPnpSGLA):
            self.X.denoise = lambda state: self.denoiser(
                state, self.X.reg_coef * self.X.epsilon**0.5
            )
            self.X.grad = lambda state: self.H.adjoint(self.Hx - self.y) / self.sigma2
        else:
            raise ValueError("Kernel type not yet supported by this model.")

    def _on_states_updated(self):
        self.Hx = self.H.forward(self.X.state)

    def update(self, rng: np.random.Generator | torch.Generator):
        """Global update of the model. Updates every kernel used by the model and computes annex variables.

        Parameters
        ----------
        rng : np.random.Generator | torch.Generator
            Random number generator, given by the sampler.
        """

        self.X.mc_step(rng)

        # update cached buffer related to X
        self.Hx = self.H.forward(self.X.state)

    def compute_potential(self) -> float:
        """Compute the potential of the likelihood for the current step.

        Returns
        -------
        float
            Potential of the targeted distribution.
        """
        return (0.5 / self.sigma2) * xp.sum((self.y.state - self.Hx) ** 2)


class DistributedGaussianDeconvolutionPnpModel(
    GaussianDeconvolutionPnpModel,
    BaseDistributedModel,
): ...
