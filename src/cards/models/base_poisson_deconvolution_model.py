r"""Base class to define Poisson deconvolution models leveraging approximate data augmentation (AXDA :cite:`Vono2020`)."""

# TODO: documentation

from abc import abstractmethod
from dataclasses import dataclass

from cards.backend import xp
from cards.estimator.mmse_builder import MMSEBuilder
from cards.models.base_model import BaseModel
from cards.operators.dft_convolution import DftConvolution
from cards.operators.mpi_dft_convolution import MpiDftConvolution
from cards.transition_kernel.base_transition_kernel import (
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
    def __init__(
        self,
        params: PoissonDeconvolutionParameters,
        convolution_operator: DftConvolution | MpiDftConvolution,
        X: BaseTransitionKernel,
        Z1: BaseTransitionKernel,
        Z2: BaseTransitionKernel,
    ):
        self.X = X
        self.Z1 = Z1
        self.Z2 = Z2
        self.convolution_operator = convolution_operator
        super().__init__()

        self.observations = params.observations

        # model hyperparameters
        self.reg_coeff = params.reg_coeff
        self.split_coef1 = params.split_coef1
        self.split_coef2 = params.split_coef2
        self.dynamic_range = params.dynamic_range

        # internal buffers
        self.convX = xp.zeros_like(self.observations)

        self.estimator_builder = MMSEBuilder(
            X.current_state.shape, dtype=X.current_state.dtype, name="X"
        )

        self.set_conditionals()

    @abstractmethod
    def set_conditionals(self):
        pass

    def aggregate_states(self):
        self.estimator_builder.aggregate_states(self.X.current_state)

    def _get_estimator_builder_states(self) -> xp.ndarray:
        # TODO: check if this instruction is absolutely needed
        if isinstance(self.X, BaseGpuTransitionKernel):
            return self.estimator_builder.estimator.get()
        return self.estimator_builder.estimator
