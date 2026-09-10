r"""Define base classes and data structures for inpainting under Gaussian noise models.

This module provides the foundational components for solving inverse problems involving
inpainting under additive white Gaussian noise. It includes the parameter
configuration data class and the abstract base model from which specific inpainting
samplers inherit.
"""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from abc import abstractmethod
from dataclasses import dataclass

from cards.core.variable import Variable
from cards.models.base_model import BaseModel
from cards.operators.distributed_masking import DistributedMasking
from cards.operators.masking import Masking
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel


@dataclass
class GaussianInpaintingParameters:
    r"""Configuration parameters for inpainting under Gaussian noise models.

    This data class encapsulates the known variables and hyperparameters required to
    define the forward model of this inverse problem.
    """

    sigma2: float
    r"""The variance of the additive white Gaussian noise, :math:`\sigma^2`."""

    reg_coeff: float
    r"""The regularization coefficient used to weight the prior relative to the likelihood."""


class BaseGaussianInpaintingModel(BaseModel):
    r"""Abstract base class for inpainting under Gaussian noise models.

    This class extends :class:`BaseModel` to provide a standard framework for
    inpainting tasks. It stores the forward operator, initializes internal
    buffers for masking operations, and retains the model hyperparameters.

    Parameters
    ----------
    estimators : list[BaseEstimator]
        A list of estimator builders used to compute parameter estimates during sampling.
    params : GaussianInpaintingParams
        The configuration parameters containing the observations, kernel, and noise variance.
    masking_operator : Masking
        The operator handling masking operations.
    X : BaseTransitionKernel
        The transition kernel responsible for sampling the primary target variable.
    """

    def __init__(
        self,
        params: GaussianInpaintingParameters,
        masking_operator: Masking | DistributedMasking,
        y: Variable,
        X: BaseTransitionKernel,
        *other_kernels: BaseTransitionKernel,
    ):
        super().__init__(X.var, *[k.var for k in other_kernels])
        self.params = params

        self.reg_coeff = params.reg_coeff
        self.sigma2 = params.sigma2

        self.H = masking_operator

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
