from abc import abstractmethod
from dataclasses import dataclass

from mcmc.backend import xp
from mcmc.estimator.estimator_builder import EstimatorBuilder
from mcmc.models.base_model import BaseModel
from mcmc.operators.dft_convolution import DftConvolution
from mcmc.operators.mpi_dft_convolution import MpiDftConvolution
from mcmc.transition_kernel.base_transition_kernel import (
    BaseGpuTransitionKernel,
    BaseTransitionKernel,
)


@dataclass
class GaussianDeconvolutionParams:
    observations: xp.ndarray
    kernel: xp.ndarray
    sigma2: float
    reg_coeff: float


class BaseGaussianDeconvolutionModel(BaseModel):
    convolution_operator: DftConvolution | MpiDftConvolution

    def __init__(self, params: GaussianDeconvolutionParams, X: BaseTransitionKernel):
        super().__init__()
        self.observations = params.observations
        self.convolution_kernel = params.kernel
        self.X = X
        self.reg_coeff = params.reg_coeff

        self.sigma2 = params.sigma2

        self.convX = xp.zeros_like(self.observations)

        self.estimator_builder = EstimatorBuilder(
            {"X": (X.current_state.shape, X.current_state.dtype)}
        )

        self.set_conditionals()

    @abstractmethod
    def set_conditionals(self): ...

    def aggregate_states(self):
        self.estimator_builder.aggregate_states({"X": self.X.current_state})

    def _get_estimator_builder_states(self) -> dict[str, xp.ndarray]:
        if isinstance(self.X, BaseGpuTransitionKernel):
            return {k: v.get() for k, v in self.estimator_builder.estimator.items()}
        return self.estimator_builder.estimator
