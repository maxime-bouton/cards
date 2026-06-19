r"""Base class defining Gaussian deconvolution models."""

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
from cards.operators.dft_convolution import DftConvolution
from cards.operators.mpi_dft_convolution import MpiDftConvolution
from cards.transition_kernel.base_transition_kernel import (
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
    def __init__(
        self,
        params: GaussianDeconvolutionParams,
        convolution_operator: DftConvolution | MpiDftConvolution,
        X: BaseTransitionKernel,
    ):
        self.X = X
        super().__init__()

        self.observations = params.observations
        self.convolution_operator = convolution_operator

        # model hyperparameters
        self.reg_coeff = params.reg_coeff
        self.sigma2 = params.sigma2

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
