r"""Linear operators.

This package provides the abstract operator class and associated linear operator examples investigated in the paper.

All operators can be run either on CPU or GPU, and admit both a serial and a distributed implementation (whenever required).

Classes
-------
:class:`~cards.operators.linear_operator.LinearOperator`
    Abstract class underlying all linear operators in the library.
:class:`~cards.operators.dft_convolution.DftConvolution`
    Serial implementation of an FFT-based convolution operator.
:class:`~cards.operators.distributed_dft_convolution.DistributedDftConvolution`
    Distributed implementation of an FFT-based convolution operator.
:class:`~cards.operators.gradient.Gradient2d`
    Serial implementation of the 2D discrete gradient operator.
:class:`~cards.operators.distributed_gradient.DistributedGradient2d`
    Distributed implementation of the 2D discrete gradient operator.
:class:`~cards.operators.distributed_torch_convolution.DistributedTorchConvolution`
    Distributed implementation of a pytorch linear convolution operator.
:class:`~cards.operators.masking.Masking`
    Implementation of the masking operator involved in inpainting problems.
"""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from .dft_convolution import DftConvolution
from .distributed_dft_convolution import DistributedDftConvolution
from .distributed_gradient import DistributedGradient2d
from .distributed_torch_convolution import DistributedTorchConvolution
from .gradient import Gradient2d
from .linear_operator import LinearOperator
from .masking import Masking

__all__ = [
    "DftConvolution",
    "DistributedDftConvolution",
    "DistributedGradient2d",
    "DistributedTorchConvolution",
    "Gradient2d",
    "LinearOperator",
    "Masking",
]
