r"""Abstract communicator class to exchange sub-arrays within a Cartesian grid
of processes with an arbitrary number of axes.
The class underlies all the computations conducted within the distributed operators implemented in :mod:`~card`.operators`.
"""

# TODO: keep Tuple[int, ...] for all shapes, convert to numpy arrays only internally (and temporarily)
# TODO: check typing (xp.ndarray or np.ndarray)

from abc import ABC, abstractmethod

import numpy as np
from mpi4py import MPI

from cards.communicators.mpi_utils import get_ranknd
from cards.slicers.cartesian_comm_slicer import CartesianCommSlicer


class BaseCartesianCommunicator(ABC):
    r"""Abstract communicator class to exchange sub-arrays across a Cartesian grid of processes with an arbitrary number of axes.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        MPI communicator.
    grid_size : numpy.ndarray[int]
        Number of processes along each axis of the Cartesian grid, as returned by ``np.array(MPI.Compute_dims(size, ndims), dtype="i")``.
    buffer_size : numpy.ndarray[int], of size ``d``
        Shape of the ``d``-dimensional buffer decomposed over the Cartesian grid of workers considered. The number of elements handled by the
        current process is computed by an instance of the
        :class:`cards.slicer.cartesian_comm_slicer.CartesianCommSlicer`
        class.
    send_size : numpy.ndarray[int], of size ``d``
        Extent of the ghost cell to be sent to contiguous facets along each axis of the Cartesian grid.
    recv_size : numpy.ndarray[int], of size ``d``
        Extent of the ghost cell to be received from contiguous facets along each axis of the Cartesian grid.
    dtype : type, optional
        Type of the buffer over which the communicator is defined, by default np.float64. The `type` is required to define sub-arrays using `MPI.Datatype`.
    backward : bool, optional
        Direction of the overlap between contiguous facets along each axis of the Cartesian grid (forward or backward overlap), by default `True`.
    tile_range : numpy.ndarray[int] or None, optional
        Index of the elements from the global array exclusively handled by the current process, defining a subarray. By default `None`, corresponding to an even tessellation of an array across the workers.

    Attributes
    ----------
    comm : mpi4py.MPI.Comm
            Underlying MPI communicator.
    grid_size : numpy.ndarray[int]
        Shape of the communication grid along each axis of the problem, as
        returned by ``np.array(MPI.Compute_dims(size, ndims), dtype="i")``.
    buffer_size : numpy.ndarray[int], of size ``d``
        Shape of the ``d``-dimensional buffer decomposed over the Cartesian grid of workers considered. The number of elements handled by the
        current process is computed by an instance of the
        :class:`cards.slicer.cartesian_comm_slicer.CartesianCommSlicer` class.
    send_size : numpy.ndarray[int], of size ``d``
        Extent of the ghost cell to be sent to contiguous facets along each axis of the Cartesian grid.
    recv_size : numpy.ndarray[int], of size ``d``
        Extent of the ghost cell to be received from contiguous facets along each axis of the Cartesian grid.
    dtype : type, optional
        Type of the buffer over which the communicator is defined, by default np.float64. The ``dtype`` is required to define sub-arrays using `MPI.Datatype`.
    backward : bool, optional
        Direction of the overlap between contiguous facets along each axis of the Cartesian grid (forward or backward overlap), by default True.
    ndims : int
        Number of axes of the arrays to be exchanged.
    rank : int
        Linear rank of the process.
    ranknd : numpy.ndarray[int]
        nD rank of the process.
    cartslicer : cards.slicer.cartesian_comm_slicer.CartesianCommSlicer
        Slicer used to define and extract messages received to / sent from the
        current worker.

    Raises
    ------
    ValueError
        `grid_size` and `buffer_size` must contain the same number of axes.
    ValueError
        `send_size` and `recv_size` must contain the same number of axes.

    Methods
    -------
    update_borders(local_array):
        Send and receive data from a given buffer.

    Note
    ----
    The following virtual methods need to be implemented in any daughter class:

    - :meth:`cards.communicators.base_cartesian_communicator.BaseCartesianCommunicator._setup_communications`
    - :meth:`cards.communicators.base_cartesian_communicator.BaseCartesianCommunicator._update_borders`
    - :meth:`cards.communicators.base_cartesian_communicator.BaseCartesianCommunicator._remove`
    """

    def __init__(
        self,
        comm: MPI.Comm,
        grid_size: np.ndarray,
        buffer_size: np.ndarray,
        send_size: np.ndarray,
        recv_size: np.ndarray,
        dtype: type = np.float64,
        backward: bool = True,
        tile_range: np.ndarray | None = None,
    ) -> None:
        self.comm = comm
        self.grid_size = grid_size
        self.dtype = dtype
        self.backward = backward

        if not self.grid_size.size == buffer_size.size:
            raise ValueError(
                "`grid_size` and `buffer_size` must contain the same number of axes."
            )
        self.buffer_size = buffer_size

        if not send_size.size == recv_size.size:
            raise ValueError(
                "`send_size` and `recv_size` must contain the same number of axes."
            )
        self.send_size = send_size
        self.recv_size = recv_size

        self.ndims = grid_size.size
        self.rank = self.comm.Get_rank()
        self.ranknd = get_ranknd(self.rank, self.grid_size)
        # self.circular_boundaries = False
        # self.cartcomm = comm.Create_cart(
        #     dims=self.grid_size,
        #     periods=self.grid_size.size * [self.circular_boundaries],
        #     reorder=False,
        # )
        # self.ranknd = np.array(self.cartcomm.Get_coords(self.rank), dtype="i")

        self.cartslicer = CartesianCommSlicer(
            self.ranknd,
            self.grid_size,
            self.buffer_size,
            self.send_size,
            self.recv_size,
            backward=self.backward,
            tile_range=tile_range,
        )

        # ? update send_size / recv_size from sync_cartesian_communicator
        # self.send_size = self.cartslicer.send_size.copy()
        # self.recv_size = self.cartslicer.recv_size.copy()

        # configure MPI-based communications
        self._setup_communications()

    @abstractmethod
    def _setup_communications(self) -> None:  # pragma: no cover
        r"""Setup all auxiliary variables and types to define
        the communications with MPI.
        """

    @abstractmethod
    def update_borders(self, local_array) -> None:  # pragma: no cover
        r"""Send and receive data from a given buffer."""

    @abstractmethod
    def remove(self) -> None:  # pragma: no cover
        r"""Trigger object finalizer to clean up auxiliary quantities when the object can be safely deleted."""
