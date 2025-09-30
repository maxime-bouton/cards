"""Implement a model used to build a solution to an inpainting problem under gaussian noise.
Can be executed on cpu or gpu depending on the settings of the backend.py file.
"""

from abc import abstractmethod
from dataclasses import dataclass

from mcmc.backend import xp
from mcmc.estimator.mmse_builder import mmse_builder
from mcmc.models.base_model import BaseModel
from mcmc.transition_kernel.base_transition_kernel import (
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
        super().__init__()
        self.observations = params.observations
        self.mask = params.mask
        self.X = X
        self.reg_coeff = params.reg_coeff

        self.sigma2 = params.sigma2

        self.estimator_builder = mmse_builder(X.current_state.shape)

        self.set_conditionals()

    @abstractmethod
    def set_conditionals(self): ...

    def aggregate_states(self):
        self.estimator_builder.aggregate_states(self.X.current_state)

    def _get_estimator_builder_states(self) -> dict[str, xp.ndarray]:
        if isinstance(self.X, BaseGpuTransitionKernel):
            return self.estimator_builder.estimator.get()
        return self.estimator_builder.estimator
