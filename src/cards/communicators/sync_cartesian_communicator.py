r"""Communicator class for synchronous communications on a Cartesian grid of
MPI processes in arbitrary dimension.
"""

# TODO: check typing (xp.ndarray or np.ndarray)

import weakref

import numpy as np
from mpi4py import MPI

from cards.communicators.base_cartesian_communicator import BaseCartesianCommunicator
from cards.communicators.mpi_utils import (
    free_custom_mpi_types,
    mpi_create_subarray_type,
)


def send_rank(ranknd, grid_size, backward: bool = True, circular: bool = False):
    r"""Identify rank of destination worker for a given communication (sync. comm.).

    Identify the linear rank of the destination worker for a given communication (synchronous communications only).

    Parameters
    ----------
    ranknd : numpy.ndarray[int]
        Multi-dimensional rank of the current process in the nD Cartesian grid
        of MPI processes.
    grid_size : numpy.ndarray[int]
        Number of processes along each dimension of the nD Cartesian grid.
    backward : bool, optional
        Direction of the overlap in the Cartesian grid along all the
        dimensions (forward or backward overlap), by default True.
    circular : bool, optional
        Consider a circular communication pattern across each axis of the
        Cartesian grid of workers to determine rank of destination workers, by
        default False.

    Returns
    -------
    dest : numpy.ndarray[int]
        Linear rank of the destination workers. The value MPI.PROC_NULL is used to encode invalid destination ranks (`MPI.PROC_NULL=-2` currently in openMPI).

    Note
    ----
    The workers at the source of a communication can be obtained by
    >>> dest = send_rank_sync(ranknd, grid_size, backward=backward)
    >>> src = send_rank_sync(ranknd, grid_size, backward=not backward)
    """

    # ndims = np.sum(grid_size > 1)
    ndims = grid_size.size
    dest = np.empty(ndims, dtype="i")
    nbr_ranknd = ranknd.copy()

    if backward:
        increment = 1
    else:
        increment = -1

    if circular:
        for axis in range(ndims):
            nbr_ranknd[axis] = (nbr_ranknd[axis] + increment) % grid_size[axis]
            dest[axis] = np.ravel_multi_index(
                nbr_ranknd,
                grid_size,
            )
            nbr_ranknd[axis] = ranknd[axis]
    else:
        # NOTE: returning the MPI.PROC_NULL index for any invalid communication axis
        for axis in range(ndims):
            nbr_ranknd[axis] += increment

            if np.any(np.bitwise_or(nbr_ranknd < 0, nbr_ranknd > grid_size - 1)):
                dest[axis] = MPI.PROC_NULL
            else:
                dest[axis] = np.ravel_multi_index(
                    nbr_ranknd,
                    grid_size,
                )
            nbr_ranknd[axis] = ranknd[axis]

    return dest


class SyncCartesianCommunicator(BaseCartesianCommunicator):
    r"""Cartesian communicator for synchronous (non-circular) communications.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Underlying MPI communicator.
    grid_size : numpy.ndarray[int]
        Shape of the communication grid along each axis of the problem, as
        returned by ``np.array(MPI.Compute_dims(size, ndims), dtype="i")``.
    buffer_size : numpy.ndarray[int], of size ``d``
        Size of the ``d`` dimensional buffer decomposed over the Cartesian grid of workers considered. The number of elements handled by the
        current process is computed by an instance of the
        :class:`cards.slicer.cartesian_comm_slicer.CartesianCommSlicer`
        class.
    send_size : numpy.ndarray[int], of size ``d``
        Size of the buffer to be sent to a neighbor worker.
    recv_size : numpy.ndarray[int], of size ``d``
        Size of the buffer to be received on the current worker.
    dtype : type, optional
        Type of the buffer over which the communicator is defined (required
        to define sub-arrays), by default np.float64. For now, restricted
        to a ``type`` from numpy.
    backward : bool, optional
        Direction of the overlap in the Cartesian grid along all the
        dimensions (forward or backward overlap), by default True.
    tile_range : numpy.ndarray[int] or None, optional
        Index of the elements from the global array exclusively handled by the current process, defining a subarray. By default None, so that it is directly specified by the object itself, dividing the global array evenly across the different workers.

    Attributes
    ----------
    ndims_comm : int
        Number of dimensions of the problem across which communications occur.
    ncomms : int
        Number of communications performed by the current worker.
    src : numpy.ndarray[int]
        Linear rank of workers from which the current process receives data.
    dest : numpy.ndarray[int]
        Linear rank of workers to which the current process sends data.
    resizedsendsubarray : list[mpi4py.MPI.Datatype | None]
        Resized MPI datatype corresponding to the subarrays sent from the
        current process, with `None` value corresponding to invalid
        communications.
    resizedrecvsubarray : list[mpi4py.MPI.Datatype | None]
        Resized MPI datatype corresponding to the subarrays received by the
        current process, with `None` value corresponding to invalid
        communications.
    _finalizer : weakref.finalize
        Finalizer object to deallocate the resources assigned to custom subarray data-types.

    Methods
    -------
    removed()
        Check whether the object has been finalized.
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
        super(SyncCartesianCommunicator, self).__init__(
            comm,
            grid_size,
            buffer_size,
            send_size,
            recv_size,
            dtype=dtype,
            backward=backward,
            tile_range=tile_range,
        )

        self.ndims_comm = self.ndims  # int(np.sum(self.grid_size > 1))
        self.ncomms = self.ndims  # int(np.sum(self.grid_size > 1))

        self._finalizer = weakref.finalize(
            self,
            free_custom_mpi_types,
            self.resizedsendsubarray,
            self.resizedrecvsubarray,
        )

    def _setup_communications(self) -> None:
        r"""Setup all auxiliary variables and types to define
        the communications with MPI.
        """
        # NOTE: entries in cartslicer.send_size and cartslicer.recv_size are
        # set to zero on workers for which no communication is possible along
        # a given dimension (i.e., at the border of the communicator it
        # circular boundaries not used)
        # ! MPI.PROC_NULL value used for invalid src/dest rank to accommodate Sendrecv instructions
        self.dest = send_rank(self.ranknd, self.grid_size, backward=self.backward)
        # ! forces MPI.PROC_NULL value for communications with 0 overlap size
        self.dest[self.cartslicer.send_size <= 0] = MPI.PROC_NULL

        self.resizedsendsubarray = mpi_create_subarray_type(
            self.cartslicer.facet_size,
            self.dest,
            self.cartslicer.starts_send,
            self.cartslicer.subsizes_send,
            dtype=self.dtype,
        )

        self.src = send_rank(self.ranknd, self.grid_size, backward=not self.backward)
        # ! forces MPI.PROC_NULL value for communications with 0 overlap size
        self.src[self.cartslicer.recv_size <= 0] = MPI.PROC_NULL

        self.resizedrecvsubarray = mpi_create_subarray_type(
            self.cartslicer.facet_size,
            self.src,
            self.cartslicer.starts_recv,
            self.cartslicer.subsizes_recv,
            dtype=self.dtype,
        )

    # NOTE: directly maintain the array to be updated within the communicator?
    # TODO: add error if local_array.dtype different from self.dtype
    def update_borders(self, local_array) -> None:
        r"""Update the borders of a local array using the communication scheme
        defined in the communicator.

        Parameters
        ----------
        local_array : numpy.ndarray, with ``d`` dimensions, float entries
            Local array to be updated through communications.

        Note
        ----
        The input array is updated in-place.
        """
        assert np.allclose(
            np.array(local_array.shape, dtype="i"), self.cartslicer.facet_size
        ), "Shape of array should be equal to self.cartslicer.facet_size"
        for d in range(self.ncomms):
            self.comm.Sendrecv(
                [local_array, 1, self.resizedsendsubarray[d]],
                self.dest[d],
                recvbuf=[local_array, 1, self.resizedrecvsubarray[d]],
                source=self.src[d],
            )

    def remove(self) -> None:
        return self._finalizer()

    @property
    def removed(self):
        r"""Check whether the object has been finalized."""
        return not self._finalizer.alive
