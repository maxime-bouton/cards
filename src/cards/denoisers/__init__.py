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
    Serial DDFB network :cite:p:`Repetti2022eusipco`.
:class:`~cards.denoisers.distributed_ddfb.DistributedDDFB`
    Distributed implementation for the Deep Dual Forward-Backward (DDFB)
    denoiser :cite:p:`Repetti2022eusipco`.
:class:`~cards.denoisers.serial_drunet.SerialDRUNet`
    Serial DRUNet network :cite:p:`Zhang2021`.
:class:`~cards.denoisers.distributed_drunet.DistributedDRUNet`
    Distributed implementation of the DRUNet :cite:p:`Zhang2021` network.
:class:`~cards.denoisers.serial_dncnn.SerialDnCNN`
    Serial DnCNN network :cite:p:`Zhang2017`.
:class:`~cards.denoisers.distributed_dncnn.DistributedDnCNN`
    Distributed implementation of the DnCNN :cite:p:`Zhang2017` network.

Examples
--------
>>> #TODO: add example usage of the models here
"""

from .base_denoiser import BaseDenoiser, BaseDistributedDenoiser
from .distributed_ddfb import DistributedDDFB
from .distributed_dncnn import DistributedDnCNN
from .distributed_drunet import DistributedDRUNet
from .serial_ddfb import SerialDDFB
from .serial_dncnn import SerialDnCNN
from .serial_drunet import SerialDRUNet

__all__ = [
    "BaseDenoiser",
    "BaseDistributedDenoiser",
    "DistributedDDFB",
    "DistributedDnCNN",
    "DistributedDRUNet",
    "SerialDDFB",
    "SerialDnCNN",
    "SerialDRUNet",
]
