from abc import abstractmethod
from dataclasses import dataclass

from mcmc.backend import xp
from mcmc.estimator.mmse_builder import mmse_builder
from mcmc.models.base_model import BaseModel
from mcmc.operators.dft_convolution import DftConvolution
from mcmc.operators.mpi_dft_convolution import MpiDftConvolution
from mcmc.transition_kernel.base_transition_kernel import (
    BaseGpuTransitionKernel,
    BaseTransitionKernel,
)


@dataclass
class PoissonDeconvolutionParameters:
    observations: xp.ndarray
    kernel: xp.ndarray
    dynamic_range: float
    reg_coeff: float
    split_coef1: float
    split_coef2: float


class BasePoissonDeconvolutionModel(BaseModel):
    convolution_operator: DftConvolution | MpiDftConvolution

    def __init__(
        self,
        params: PoissonDeconvolutionParameters,
        X: BaseTransitionKernel,
        Z1: BaseTransitionKernel,
        Z2: BaseTransitionKernel,
    ):
        super().__init__()
        self.observations = params.observations
        self.convolution_kernel = params.kernel
        self.X = X
        self.Z1 = Z1
        self.Z2 = Z2
        self.reg_coeff = params.reg_coeff

        self.split_coef1 = params.split_coef1
        self.split_coef2 = params.split_coef2

        self.dynamic_range = params.dynamic_range

        self.convX = xp.zeros_like(self.observations)

        self.estimator_builder = mmse_builder(X.current_state.shape)

        self.set_conditionals()

    @abstractmethod
    def set_conditionals(self): ...

    def aggregate_states(self):
        self.estimator_builder.aggregate_states(self.X.current_state)

    def _get_estimator_builder_states(self) -> xp.ndarray:
        if isinstance(self.X, BaseGpuTransitionKernel):
            return self.estimator_builder.estimator.get()
        return self.estimator_builder.estimator
