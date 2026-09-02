r"""Base class to define Poisson deconvolution models leveraging approximate data augmentation (AXDA :cite:`Vono2021`)."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

# TODO: update documentation

from abc import abstractmethod
from dataclasses import dataclass

from cards.core.variable import Variable
from cards.models.base_model import BaseModel
from cards.operators.dft_convolution import DftConvolution
from cards.operators.distributed_dft_convolution import DistributedDftConvolution
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel


@dataclass
class PoissonDeconvolutionParams:
    r"""Configuration parameters for deconvolution under Gaussian noise models.

    This data class encapsulates the known variables and hyperparameters required to
    define the forward model of this inverse problem.
    """

    dynamic_range: float
    r"""The dynamic range of the ground truth image."""

    reg_coeff: float
    r"""The regularization coefficient used to weight the prior relative to the likelihood."""

    split_coef1: float
    r"""AXDA splitting parameter (within the likelihood)."""

    split_coef2: float
    r"""AXDA splitting parameter (within the prior)."""


class BasePoissonDeconvolutionModel(BaseModel):
    r"""Abstract base class for deconvolution under Poisson noise models.

    This class extends :class:`BaseModel` to provide a standard framework for
    deconvolution tasks. It stores the forward operator, initializes internal
    buffers for convolution operations, and retains the model hyperparameters.

    Parameters
    ----------
    estimators : list[BaseEstimator]
        A list of estimator builders used to compute parameter estimates during sampling.
    params : GaussianDeconvolutionParams
        The configuration parameters containing the observations, kernel, and noise variance.
    convolution_operator : DftConvolution | DistributedDftConvolution
        The operator handling the forward and adjoint convolution operations.
    X : BaseTransitionKernel
        The transition kernel responsible for sampling the primary target variable.
    """

    def __init__(
        self,
        params: PoissonDeconvolutionParams,
        convolution_operator: DftConvolution | DistributedDftConvolution,
        y: Variable,
        X: BaseTransitionKernel,
        *other_kernels: BaseTransitionKernel,
        # Z1: BaseTransitionKernel,
        # Z2: BaseTransitionKernel,
    ):
        super().__init__(X.var, *[k.var for k in other_kernels])
        self.params = params

        self.reg_coeff = params.reg_coeff
        self.dynamic_range = params.dynamic_range

        self.H = convolution_operator

        self.y = y
        self.X = X

        self.Hx = self.H.forward(self.X.state)

        # self.Z1 = Z1
        # self.Z2 = Z2

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
