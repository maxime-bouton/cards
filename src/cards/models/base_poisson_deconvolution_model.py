r"""Base class to define Poisson deconvolution models leveraging approximate data augmentation (AXDA :cite:`Vono2021`)."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

# TODO: documentation

from abc import abstractmethod
from dataclasses import dataclass

import cards.backend as xp
from cards.estimators.base_estimator import BaseEstimator
from cards.models.base_model import BaseModel
from cards.operators.dft_convolution import DftConvolution
from cards.operators.distributed_dft_convolution import DistributedDftConvolution
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel


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
        estimators: list[BaseEstimator],
        params: PoissonDeconvolutionParameters,
        convolution_operator: DftConvolution | DistributedDftConvolution,
        X: BaseTransitionKernel,
        Z1: BaseTransitionKernel,
        Z2: BaseTransitionKernel,
    ):
        self.X = X
        self.Z1 = Z1
        self.Z2 = Z2
        self.convolution_operator = convolution_operator
        super().__init__(estimators)

        self.observations = params.observations

        # model hyperparameters
        self.reg_coeff = params.reg_coeff
        self.split_coef1 = params.split_coef1
        self.split_coef2 = params.split_coef2
        self.dynamic_range = params.dynamic_range

        # internal buffers
        self.convX = xp.zeros_like(self.observations)

        self.set_conditionals()

    @abstractmethod
    def set_conditionals(self):
        pass
