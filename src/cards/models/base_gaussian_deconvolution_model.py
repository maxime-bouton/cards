r"""Define base classes and data structures for deconvolution under Gaussian noise models.

This module provides the foundational components for solving inverse problems involving
deconvolution under additive white Gaussian noise. It includes the parameter
configuration data class and the abstract base model from which specific deconvolution
samplers inherit.
"""

from abc import abstractmethod
from dataclasses import dataclass

from cards.core.variable import Variable
from cards.models.base_model import BaseModel
from cards.operators.dft_convolution import DftConvolution
from cards.operators.mpi_dft_convolution import MpiDftConvolution
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel


@dataclass
class GaussianDeconvolutionParams:
    r"""Configuration parameters for deconvolution under Gaussian noise models.

    This data class encapsulates the known variables and hyperparameters required to
    define the forward model of this inverse problem.
    """

    sigma2: float
    r"""The variance of the additive white Gaussian noise, :math:`\sigma^2`."""

    reg_coeff: float
    r"""The regularization coefficient used to weight the prior relative to the likelihood."""


class BaseGaussianDeconvolutionModel(BaseModel):
    r"""Abstract base class for deconvolution under Gaussian noise models.

    This class extends :class:`BaseModel` to provide a standard framework for
    deconvolution tasks. It stores the forward operator, initializes internal
    buffers for convolution operations, and retains the model hyperparameters.

    Parameters
    ----------
    estimators : list[BaseEstimatorBuilder]
        A list of estimator builders used to compute parameter estimates during sampling.
    params : GaussianDeconvolutionParams
        The configuration parameters containing the observations, kernel, and noise variance.
    convolution_operator : DftConvolution | MpiDftConvolution
        The operator handling the forward and adjoint convolution operations.
    X : BaseTransitionKernel
        The transition kernel responsible for sampling the primary target variable.
    """

    def __init__(
        self,
        params: GaussianDeconvolutionParams,
        convolution_operator: DftConvolution | MpiDftConvolution,
        y: Variable,
        X: BaseTransitionKernel,
        *other_kernels: BaseTransitionKernel,
    ):
        super().__init__(X.var, *[k.var for k in other_kernels])
        self.params = params

        self.reg_coeff = params.reg_coeff
        self.sigma2 = params.sigma2

        self.H = convolution_operator

        self.y = y
        self.X = X

        self.Hx = self.H.forward(self.X.state)

    @abstractmethod
    def set_conditionals(self) -> None:
        r"""Set up the conditional distributions for the transition kernels.

        This method is called automatically at the end of initialization.
        Inheriting classes must implement this to define how the transition
        kernels update their respective variables based on the specific
        inference algorithm (e.g., Gibbs sampling or PnP-ULA).
        """

    def compile(self) -> None:
        self.set_conditionals()
