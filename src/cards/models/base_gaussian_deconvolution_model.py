r"""Define base classes and data structures for deconvolution under Gaussian noise models.

This module provides the foundational components for solving inverse problems involving
deconvolution under additive white Gaussian noise. It includes the parameter
configuration data class and the abstract base model from which specific deconvolution
samplers inherit.
"""

from abc import abstractmethod
from dataclasses import dataclass

import cards.backend as xp
from cards.estimators.base_estimator import BaseEstimator
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

    observations: xp.ndarray
    r"""The observed degraded signal or image, often denoted as :math:`\mathbf{y}`."""

    # kernel: xp.ndarray
    # r"""The blur kernel defining the convolution operator."""

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
        estimators: list[BaseEstimator],
        params: GaussianDeconvolutionParams,
        convolution_operator: DftConvolution | MpiDftConvolution,
        X: BaseTransitionKernel,
    ):
        self.X = X
        super().__init__(estimators)

        self.observations = params.observations
        self.convolution_operator = convolution_operator

        # model hyperparameters
        self.reg_coeff = params.reg_coeff
        self.sigma2 = params.sigma2

        # internal buffers
        self.convX = xp.zeros_like(self.observations)

        self.set_conditionals()

    @abstractmethod
    def set_conditionals(self) -> None:
        r"""Set up the conditional distributions for the transition kernels.

        This method is called automatically at the end of initialization.
        Inheriting classes must implement this to define how the transition
        kernels update their respective variables based on the specific
        inference algorithm (e.g., Gibbs sampling or PnP-ULA).
        """
