r"""Implementation of a Gaussian inpainting model using a PnP prior to reproduce the experiments reported in :cite:p:`Bouton2026`."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

# TODO: documentation

from dataclasses import dataclass

import numpy as np
import torch

import cards.backend as xp
from cards.core.variable import Variable
from cards.denoisers.base_denoiser import BaseDenoiser
from cards.models.base_gaussian_inpainting_model import (
    BaseGaussianInpaintingModel,
    GaussianInpaintingParameters,
)
from cards.models.base_model import BaseDistributedModel
from cards.operators.distributed_masking import DistributedMasking
from cards.operators.masking import Masking
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel

# from cards.transition_kernels.pnp_sgla import PnpSGLA
from cards.transition_kernels.pnp_ula import PnpULA


@dataclass
class GaussianInpaintingPnpParameters(GaussianInpaintingParameters): ...


class GaussianInpaintingPnpModel(BaseGaussianInpaintingModel):
    def __init__(
        self,
        params: GaussianInpaintingPnpParameters,
        masking_operator: Masking | DistributedMasking,
        y: Variable,
        X: BaseTransitionKernel,
        denoiser: BaseDenoiser,
    ):
        super().__init__(params, masking_operator, y, X)
        self.denoiser = denoiser

    def set_conditionals(self):
        if isinstance(self.X, PnpULA):
            self.X.denoise = lambda state: self.denoiser(
                state,
                self.X.epsilon**0.5,
            )
            self.X.project = lambda state: state.clip(-1, 2)
            # NOTE: self.Hx could be replaced by state, as current implementation of masking based on term-wise multiplication
            self.X.grad = lambda state: (
                self.H.adjoint(self.Hx - self.y.state) / self.sigma2
            )
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
        """compute_potential Computes the potential.

        Returns
        -------
        float
            Potential of the targeted law.
        """
        p = xp.sum((self.y.state - self.Hx) ** 2) / (2 * self.sigma2)
        return p


class DistributedGaussianInpaintingPnpModel(
    GaussianInpaintingPnpModel,
    BaseDistributedModel,
): ...
