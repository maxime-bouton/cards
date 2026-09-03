r"""Implementation of a Poisson deconvolution model using a TV prior to reproduce the experiments reported in :cite:p:`Bouton2026`."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

# TODO: documentation

import numpy as np
import torch

import cards.backend as xp
from cards.core.variable import Variable
from cards.functionals.prox import (
    KL,
    l21_norm,
    prox_KL,
    prox_l21norm,
    prox_nonegativity,
)
from cards.models.base_model import BaseDistributedModel
from cards.models.base_poisson_deconvolution_model import (
    BasePoissonDeconvolutionModel,
    PoissonDeconvolutionParams,
)
from cards.operators.dft_convolution import DftConvolution
from cards.operators.distributed_dft_convolution import DistributedDftConvolution
from cards.operators.distributed_gradient import DistributedGradient2d
from cards.operators.gradient import Gradient2d
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel
from cards.transition_kernels.psgla import PSGLA


class PoissonDeconvolutionTvModel(BasePoissonDeconvolutionModel):
    def __init__(
        self,
        params: PoissonDeconvolutionParams,
        convolution_operator: DftConvolution | DistributedDftConvolution,
        gradient_operator: Gradient2d | DistributedGradient2d,
        y: Variable,
        X: BaseTransitionKernel,
        Z1: BaseTransitionKernel,
        Z2: BaseTransitionKernel,
    ):
        super().__init__(params, convolution_operator, y, X, Z1, Z2)

        self.Z1 = Z1
        self.Z2 = Z2

        self.G = gradient_operator
        self.Gx = self.G.forward(self.X.state)

        self.split_coef1 = params.split_coef1
        self.split_coef2 = params.split_coef2
        self.reg_coeff = params.reg_coeff

    def set_conditionals(self) -> None:
        """Set the conditionals of the transition kernels including the coupling between those kernels."""
        if isinstance(self.X, PSGLA):
            self.X.prox = prox_nonegativity
            self.X.grad = lambda state: (
                self.dynamic_range**2
                * self.H.adjoint(self.Hx - self.Z1.state / self.dynamic_range)
                / self.split_coef1
                + self.G.adjoint(self.Gx - self.Z2.state) / self.split_coef2
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

        if isinstance(self.Z1, PSGLA):
            self.Z1.prox = lambda state: prox_KL(
                state, self.y.state, lam=self.Z1.step_size
            )
            self.Z1.grad = lambda state: (
                (state - self.dynamic_range * self.Hx) / self.split_coef1
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

        if isinstance(self.Z2, PSGLA):
            self.Z2.prox = lambda state: prox_l21norm(
                state, lam=self.Z2.step_size * self.reg_coeff
            )
            self.Z2.grad = lambda state: (state - self.Gx) / self.split_coef2
        else:
            raise ValueError("Kernel type not yet supported by this model.")

    def _on_states_updated(self):
        self.Hx = self.H.forward(self.X.state)
        self.Gx = self.G.forward(self.X.state)

    def update(self, rng: np.random.Generator | torch.Generator):
        """Gobal update of the model. Updates every kernel used by the model and computes annex variables.

        Parameters
        ----------
        rng : np.random.Generator | torch.Generator
            Random number generator, given by the sampler.
        """

        self.X.mc_step(rng)
        self.Gx = self.G.forward(self.X.state)
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
        p = KL(self.Z1.state, self.y.state)
        p += xp.sum(self.Z1.state - self.dynamic_range * self.Hx) ** 2 / (
            2 * self.split_coef1
        )
        p += xp.sum((self.Gx - self.Z2.state) ** 2) / (2 * self.split_coef2)
        p += self.reg_coeff * l21_norm(self.Z2.state)
        return p


class DistributedPoissonDeconvolutionTvModel(
    PoissonDeconvolutionTvModel,
    BaseDistributedModel,
): ...
