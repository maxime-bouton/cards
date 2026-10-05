r"""Implementation of slicer classes to support the tessellation and
communication of tensor ghost cells across a Cartesian grid of processes.

Base Classes
------------
:class:`~cards.slicers.base_comm_slicer.CommSlicer`
    Abstract class defining a minimal interface for a slicer underlying the communicator classes in CARDS.

Concrete Classes
----------------
:class:`~cards.slicers.cartesian_comm_slicer.CartesianCommSlicer`
    Specialized slicer class to define and communicate ghost cells in arbitary dimensions across a Crtesian grid of processes.

Examples
--------
>>> #TODO: add example usage of the models here
"""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from .base_comm_slicer import BaseCommSlicer
from .cartesian_comm_slicer import CartesianCommSlicer

__all__ = [
    "BaseCommSlicer",
    "CartesianCommSlicer",
]
