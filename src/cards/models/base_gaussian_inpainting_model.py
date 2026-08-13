r"""Base class defining Gaussian inpainting models."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.

# TODO: documentation

from abc import abstractmethod
from dataclasses import dataclass

import cards.backend as xp
from cards.estimators.base_estimator import BaseEstimator
from cards.models.base_model import BaseModel
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel


@dataclass
class GaussianInpaintingParameters:
    observations: xp.ndarray
    mask: xp.ndarray
    sigma2: float
    reg_coeff: float


class BaseGaussianInpaintingModel(BaseModel):
    def __init__(
        self,
        estimators: list[BaseEstimator],
        params: GaussianInpaintingParameters,
        X: BaseTransitionKernel,
    ):
        self.X = X
        super().__init__(estimators)

        self.observations = params.observations
        self.mask = params.mask

        # model hyperparameters
        self.reg_coeff = params.reg_coeff
        self.sigma2 = params.sigma2

        self.set_conditionals()

    @abstractmethod
    def set_conditionals(self):
        pass
