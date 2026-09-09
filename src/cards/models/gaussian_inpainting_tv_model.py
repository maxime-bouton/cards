r"""Implementation of a Gaussian inpainting model using a TV prior to reproduce the experiments reported in :cite:p:`Bouton2026`."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

# TODO: documentation

from dataclasses import dataclass

import numpy as np
import torch

import cards.backend as xp
from cards.core.variable import Variable
from cards.functionals.prox import l21_norm, prox_l21norm, prox_nonegativity
from cards.models.base_gaussian_inpainting_model import (
    BaseGaussianInpaintingModel,
    GaussianInpaintingParameters,
)
from cards.models.base_model import BaseDistributedModel
from cards.operators.distributed_gradient import DistributedGradient2d
from cards.operators.distributed_masking import DistributedMasking
from cards.operators.gradient import Gradient2d
from cards.operators.masking import Masking
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel
from cards.transition_kernels.psgla import PSGLA


@dataclass
class GaussianInpaintingTvParameters(GaussianInpaintingParameters):
    split_coef: float


class GaussianInpaintingTvModel(BaseGaussianInpaintingModel):
    gradient_operator: Gradient2d | DistributedGradient2d

    def __init__(
        self,
        params: GaussianInpaintingTvParameters,
        masking_operator: Masking | DistributedMasking,
        gradient_operator: Gradient2d | DistributedGradient2d,
        y: Variable,
        X: BaseTransitionKernel,
        Z: BaseTransitionKernel,
    ):
        super().__init__(params, masking_operator, y, X, Z)

        self.Z = Z
        self.G = gradient_operator
        self.Gx = self.G.forward(self.X.state)

        self.split_coef = params.split_coef
        self.reg_coeff = params.reg_coeff

    def set_conditionals(self) -> None:
        """Set the conditionals of the transition kernels including the coupling between those kernels."""
        if isinstance(self.X, PSGLA):
            self.X.prox = prox_nonegativity
            self.X.grad = lambda state: (
                self.H.adjoint(self.Hx - self.y.state) / self.sigma2
                + self.G.adjoint(self.Gx - self.Z.state) / self.split_coef
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

        if isinstance(self.Z, PSGLA):
            self.Z.prox = lambda state: prox_l21norm(
                state, lam=self.Z.step_size * self.reg_coeff, axis=0
            )
            self.Z.grad = lambda state: (state - self.Gx) / self.split_coef
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

        # update cached buffers related to X
        self.Hx = self.H.forward(self.X.state)
        self.Gx = self.G.forward(self.X.state)

        self.Z.mc_step(rng)

    def compute_potential(self) -> float:
        """compute_potential Computes the potential.

        Returns
        -------
        float
            Potential of the targeted law.
        """
        p = (0.5 / self.sigma2) * xp.sum((self.y.state - self.Hx) ** 2)
        p += xp.sum((self.Gx - self.Z.state) ** 2) / (2 * self.split_coef)
        p += self.reg_coeff * l21_norm(self.Z.state)
        return p


class DistributedGaussianInpaintingTvModel(
    GaussianInpaintingTvModel,
    BaseDistributedModel,
): ...
