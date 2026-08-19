r"""Communicator classes for MPI-based Cartesian communications.

This package provides the core classes to conduct MPI-based communications over Cartesian grids of workers of arbitary dimension.

Base Class
----------
:class:`~cards.communicators.base_cartesian_communicator.BaseCartesianCommunicator`
    Abstract communicator class to exchange sub-arrays within a Cartesian grid of processes with an arbitrary number of axes.

Concrete Implementations
------------------------
:class:`~cards.communicators.sync_cartesian_communicator.SyncCartesianCommunicator`
    Communicator class for synchronous communications on a Cartesian grid of MPI processes in arbitrary dimension.

:class:`~cards.communicators.shared_communicator.SharedCommunicator`
    Class triggering communications over a Cartesian pattern shared by a collection of distributed operators.

Examples
--------
>>> #TODO: add example usage of the models here
"""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from .base_cartesian_communicator import BaseCartesianCommunicator
from .shared_communicator import SharedCommunicator
from .sync_cartesian_communicator import SyncCartesianCommunicator

__all__ = [
    "BaseCartesianCommunicator",
    "SyncCartesianCommunicator",
    "SharedCommunicator",
]
