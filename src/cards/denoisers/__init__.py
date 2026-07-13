r"""Denoiser neural-networks used in Plug-and-Play (PnP) inference approaches.

This package provides several denoiser neural networks, implemented in serial and distributed contexts, including:
- DRUNet :cite:p:`Zhang2021` (adapted from the original `implementation of the authors on github <https://github.com/cszn/KAIR/blob/master/models/network_unet.py/>`_)
- DnCNN :cite:p:`Zhang2017` (adapted from the original `implementation of the authors on github <https://github.com/cszn/KAIR/blob/master/models/network_dncnn.py/>`_)
- DDFB :cite:p:`Repetti2022eusipco`

Base Classes
------------
:class:`~cards.denoisers.base_denoiser.BaseDenoiser`
    Generic abstract class encoding serial image denoising neural-networks.
:class:`~cards.denoisers.base_denoiser.BaseDistributedDenoiser`
    Generic abatrsct class for distributed image denoising neural-networks.

Concrete Implementations
------------------------
:class:`~cards.denoisers.serial_ddfb.SerialDDFB`

:class:`~cards.denoisers.mpi_ddfb.MpiDDFB`

:class:`~cards.denoisers.serial_drunet.SerialDRUNet`

:class:`~cards.denoisers.mpi_drunet.MpiDRUNet`

:class:`~cards.denoisers.serial_dncnn.SerialDnCNN`

:class:`~cards.denoisers.mpi_dncnn.MpiDnCNN`

Examples
--------
>>> #TODO: add example usage of the models here
"""

from .base_denoiser import BaseDenoiser, BaseDistributedDenoiser
from .mpi_ddfb import MpiDDFB
from .mpi_dncnn import MpiDnCNN
from .mpi_drunet import MpiDRUNet
from .serial_ddfb import SerialDDFB
from .serial_dncnn import SerialDnCNN
from .serial_drunet import SerialDRUNet

__all__ = [
    "BaseDenoiser",
    "BaseDistributedDenoiser",
    "MpiDDFB",
    "MpiDnCNN",
    "MpiDRUNet",
    "SerialDDFB",
    "SerialDnCNN",
    "SerialDRUNet",
]
