r"""Base class defining Gaussian inpainting models."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.

# TODO: documentation

from abc import abstractmethod
from dataclasses import dataclass

from cards.backend import xp
from cards.estimator.mmse_builder import MMSEBuilder
from cards.models.base_model import BaseModel
from cards.transition_kernel.base_transition_kernel import (
    BaseGpuTransitionKernel,
    BaseTransitionKernel,
)


@dataclass
class GaussianInpaintingParameters:
    observations: xp.ndarray
    mask: xp.ndarray
    sigma2: float
    reg_coeff: float


class BaseGaussianInpaintingModel(BaseModel):
    def __init__(self, params: GaussianInpaintingParameters, X: BaseTransitionKernel):
        self.X = X
        super().__init__()

        self.observations = params.observations
        self.mask = params.mask
        self.reg_coeff = params.reg_coeff
        self.sigma2 = params.sigma2

        self.estimator_builder = MMSEBuilder(
            X.current_state.shape, dtype=X.current_state.dtype, name="X"
        )
        self.set_conditionals()

    @abstractmethod
    def set_conditionals(self):
        pass

    def aggregate_states(self):
        self.estimator_builder.aggregate_states(self.X.current_state)

    def _get_estimator_builder_states(self) -> dict[str, xp.ndarray]:
        # TODO: check if this instruction is absolutely needed
        if isinstance(self.X, BaseGpuTransitionKernel):
            return self.estimator_builder.estimator.get()
        return self.estimator_builder.estimator
