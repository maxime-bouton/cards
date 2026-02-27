"""Abstract communicator class to exchange sub-arrays within a Cartesian grid
of processes."""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)

from abc import ABC, abstractmethod

import numpy as np

from mcmc.communicator.mpi_utils import get_ranknd
from mcmc.slicer.cartesian_comm_slicer import CartesianCommSlicer


class BaseCartesianCommunicator(ABC):
    r"""Base communicator object underlying the distributed operations
    leveraged within the samplers.

    Attributes
    ----------
    comm : mpi4py.MPI.Comm
            Underlying MPI communicator.
    grid_size : numpy.ndarray[int]
        Size of the communication grid along each axis of the problem, as
        returned by ``np.array(MPI.Compute_dims(size, ndims), dtype="i")``.
    buffer_size : numpy.ndarray[int], of size ``d``
        Number of elements along each of the ``d`` dimensions of the buffer
        handled by the current process (including data received by neigbor
        workers).
    send_size : numpy.ndarray[int], of size ``d``
        Size of the buffer to be sent to a neighbor worker.
    recv_size : numpy.ndarray[int], of size ``d``
        Size of the buffer to be received on the current worker.
    dtype : numpy.dtype, optional
        Type of the buffer over which the communicator is defined (required
        to define sub-arrays), by default np.float64. For now, restricted
        to a ``np.dtype``.
    backward : bool, optional
        Direction of the overlap in the Cartesian grid along all the
        dimensions (forward or backward overlap), by default True.
    ndims : int
        Number of axes of the arrays to be exchanged.
    rank : int
        Linear rank of the process.
    ranknd : numpy.ndarray[int]
        nD rank of the process.
    cartslicer : dsgs.experimental.slicer.cartesian_comm_slicer.CartesianCommSlicer
        Slicer used to define and extract messages received to / sent from the
        current worker.

    Note
    ----
        The following virtual methods need to be implemented in any daughter class:

        - :meth:`dsgs.experimental.communicators.base_communicator.BaseCartesianCommunicator._setup_communications`
        - :meth:`dsgs.experimental.communicators.base_communicator.BaseCartesianCommunicator._update_borders`
        - :meth:`dsgs.experimental.communicators.base_communicator.BaseCartesianCommunicator._remove`
    """

    def __init__(
        self,
        comm,
        grid_size,
        buffer_size,
        send_size,
        recv_size,
        dtype=np.float64,
        backward=True,
        tile_range=None,
    ):
        """Abstract communicator object to communicate along a Cartesian grid of workers.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            Underlying MPI communicator.
        grid_size : numpy.ndarray[int]
            Size of the communication grid along each axis of the problem, as
            returned by ``np.array(MPI.Compute_dims(size, ndims), dtype="i")``.
        buffer_size : numpy.ndarray[int], of size ``d``
            Size of the global array split across the Cartesian grid of MPI processes considered.
        send_size : numpy.ndarray[int], of size ``d``
            Size of the buffer to be sent to a neighbor worker.
        recv_size : numpy.ndarray[int], of size ``d``
            Size of the buffer to be received on the current worker.
        dtype : numpy.dtype, optional
            Type of the buffer over which the communicator is defined (required
            to define sub-arrays), by default np.float64. For now, restricted
            to a ``np.dtype``.
        backward : bool, optional
            Direction of the overlap in the Cartesian grid along all the
            dimensions (forward or backward overlap), by default True.
        tile_range : numpy.ndarray[int] or None, optional
            Index of the elements from the global array exclusively handled by the current process, defining a subarray. By default None, so that it is directly specified by the object itself, dividing the global array evenly across the different workers.

        Raises
        ------
        ValueError
            `grid_size` and `buffer_size` must contain the same number of element (same number of dimensions).
        ValueError
            `send_size` and `recv_size` must contain the same number of element (same number of dimensions).
        """
        self.comm = comm
        self.grid_size = grid_size
        self.dtype = dtype
        self.backward = backward

        # * size of buffer (including overlap)
        if not self.grid_size.size == buffer_size.size:
            raise ValueError(
                "`grid_size` and `buffer_size` must contain the same number of element (same number of dimensions)."
            )
        self.buffer_size = buffer_size

        if not send_size.size == recv_size.size:
            raise ValueError(
                "`send_size` and `recv_size` must contain the same number of element (same number of dimensions)."
            )
        self.send_size = send_size
        self.recv_size = recv_size

        # secondary attributes
        self.ndims = grid_size.size
        self.rank = self.comm.Get_rank()

        # ! only for Cartesian communicator, not needed for now
        # self.circular_boundaries = False
        # self.cartcomm = comm.Create_cart(
        #     dims=self.grid_size,
        #     periods=self.grid_size.size * [self.circular_boundaries],
        #     reorder=False,
        # )
        # self.ranknd = np.array(self.cartcomm.Get_coords(self.rank), dtype="i")
        self.ranknd = get_ranknd(self.rank, self.grid_size)

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

        # configure communications (defined with MPI)
        self._setup_communications()

    @abstractmethod
    def _setup_communications(self):  # pragma: no cover
        """Setup all auxiliary variables and types to define
        the communications with MPI.
        """
        return NotImplemented

    @abstractmethod
    def update_borders(self, local_array):  # pragma: no cover
        """Send and receive data from a given buffer.

        Parameters
        ----------
        local_array : numpy.ndarray, with ``d`` dimensions
            Local array to be updated through communications.

        Returns
        -------
        NotImplemented

        Note
        ----
        The method needs to be implemented in any class inheriting from
        BaseCommunicator.
        """
        return NotImplemented

    @abstractmethod
    def remove(self):  # pragma: no cover
        """Base function to clean up auxiliary quantities when the object can be
        safely deleted.

        Returns
        -------
        NotImplemented
        """
        return NotImplemented
