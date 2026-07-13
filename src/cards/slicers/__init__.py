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

from .base_comm_slicer import BaseCommSlicer
from .cartesian_comm_slicer import CartesianCommSlicer

__all__ = [
    "BaseCommSlicer",
    "CartesianCommSlicer",
]
