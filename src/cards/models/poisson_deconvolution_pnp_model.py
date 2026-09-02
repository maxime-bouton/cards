r"""Implementation of a Poisson deconvolution model using a PnP prior to reproduce the experiments reported in :cite:p:`Bouton2026`."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

# TODO: documentation
# TODO: typing

from dataclasses import dataclass

import numpy as np
import torch

import cards.backend as xp
from cards.core.variable import Variable
from cards.denoisers.base_denoiser import BaseDenoiser
from cards.functionals.prox import KL, prox_KL, prox_nonegativity
from cards.models.base_model import BaseDistributedModel
from cards.models.base_poisson_deconvolution_model import (
    BasePoissonDeconvolutionModel,
    PoissonDeconvolutionParams,
)
from cards.operators.dft_convolution import DftConvolution
from cards.operators.distributed_dft_convolution import DistributedDftConvolution
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel
from cards.transition_kernels.pnp_sgla import PnpSGLA
from cards.transition_kernels.pnp_ula import PnpULA
from cards.transition_kernels.psgla import PSGLA


@dataclass
class PoissonDeconvolutionPnpParams(PoissonDeconvolutionParams): ...


class PoissonDeconvolutionPnpModel(BasePoissonDeconvolutionModel):
    def __init__(
        self,
        params: PoissonDeconvolutionPnpParams,
        convolution_operator: DftConvolution | DistributedDftConvolution,
        y: Variable,
        X: BaseTransitionKernel,
        Z1: BaseTransitionKernel,
        Z2: BaseTransitionKernel,
        denoiser: BaseDenoiser,
    ):
        super().__init__(params, convolution_operator, y, X, Z1, Z2)
        self.denoiser = denoiser

        self.split_coef1 = self.params.split_coef1
        self.split_coef2 = self.params.split_coef2
        self.Z1 = Z1
        self.Z2 = Z2

    def set_conditionals(self) -> None:
        """Set the conditionals of the transition kernels including the coupling between those kernels."""
        if isinstance(self.X, PSGLA):
            self.X.prox = prox_nonegativity
            self.X.grad = lambda state: (
                self.dynamic_range**2
                * self.H.adjoint(self.Hx - self.Z1.state / self.dynamic_range)
                / self.split_coef1
                + (state - self.Z2.state) / self.split_coef2
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

        if isinstance(self.Z1, PSGLA):
            self.Z1.prox = lambda state: prox_KL(state, self.y, lam=self.Z1.step_size)
            self.Z1.grad = lambda state: (
                (state - self.dynamic_range * self.Hx) / self.split_coef1
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

        if isinstance(self.Z2, PnpULA):
            self.Z2.denoise = lambda state: self.denoiser(
                state,
                self.Z2.epsilon**0.5,
            )
            self.Z2.project = lambda state: state.clip(-1, 2)
            self.Z2.grad = lambda state: (state - self.X.state) / self.split_coef2

        if isinstance(self.Z2, PnpSGLA):
            self.Z2.denoise = lambda state: self.denoiser(
                state,
                self.Z2.reg_coef * self.Z2.epsilon**0.5,
            )
            self.Z2.grad = lambda state: (state - self.X.state) / self.split_coef2
        else:
            raise ValueError("Kernel type not yet supported by this model.")

    def _on_states_updated(self):
        self.Hx = self.H.forward(self.X.state)

    def update(self, rng: np.random.Generator | torch.Generator) -> None:
        """Global update of the model. Updates every kernel used by the model and computes annex variables.

        Parameters
        ----------
        rng : np.random.Generator | torch.Generator
            Random number generator, given by the sampler.
        """

        self.X.mc_step(rng)

        # update cached buffer related to X
        self.Hx = self.H.forward(self.X.state)

        self.Z1.mc_step(rng)
        self.Z2.mc_step(rng)

    def compute_potential(self) -> float:
        """compute_potential Computes the potential.

        Returns
        -------
        float
            Potential of the targeted law.
        """
        p = KL(self.Z1.state, self.y)
        p += xp.sum(self.Z1.state - self.dynamic_range * self.Hx) ** 2 / (
            2 * self.split_coef1
        )
        p += xp.sum((self.X.state - self.Z2.state) ** 2) / (2 * self.split_coef2)
        return p


class DistributedPoissonDeconvolutionPnpModel(
    PoissonDeconvolutionPnpModel,
    BaseDistributedModel,
): ...
