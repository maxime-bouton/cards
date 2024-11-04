"""Communicator class for synchronous communications on a Cartesian grid of
MPI processes."""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)

import weakref

import numpy as np

from mcmc.communicator.base_communicator import BaseCartesianCommunicator
from mcmc.communicator.mpi_utils import (
    free_custom_mpi_types,
    mpi_create_subarray_type,
)


# TODO : modify to accommodate circular communications (mod to be added)
# TODO: debug in circular communication mod, need to make sure more than one element along a dimension to use this option
def send_rank(ranknd, grid_size, backward=True, circular=False):
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
        Linear rank of the destination workers. The value -1 is used to encode invalid destination ranks.

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
        # return -1 index for invalid communications
        for axis in range(ndims):
            nbr_ranknd[axis] += increment

            if np.any(np.bitwise_or(nbr_ranknd < 0, nbr_ranknd > grid_size - 1)):
                dest[axis] = -1
            else:
                dest[axis] = np.ravel_multi_index(
                    nbr_ranknd,
                    grid_size,
                )
            nbr_ranknd[axis] = ranknd[axis]

    return dest


class SyncCartesianCommunicator(BaseCartesianCommunicator):
    r"""Cartesian communicator for synchronous (non-circular) communications.

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
        Finalizer object to deallocate the resources assigned to
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
        """Initializer of any SyncCartesianCommunicator object.

        Parameters
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
        tile_range : numpy.ndarray[int] or None, optional
            Index of the elements from the global array exclusively handled by the current process, defining a subarray. By default None, so that it is directly specified by the object itself, dividing the global array evenly across the different workers.
        """
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

    def _setup_communications(self):
        """Setup all auxiliary variables and types to define
        the communications with MPI.
        """
        # NOTE: entries in cartslicer.send_size and cartslicer.recv_size are
        # set to zero on workers for which no communication is possible along
        # a given dimension (i.e., at the border of the communicator it
        # circular boundaries not used)
        # ! -1 value used for invalid src/dest rank to accommodate Sendrecv instructions
        self.dest = send_rank(self.ranknd, self.grid_size, backward=self.backward)

        self.resizedsendsubarray = mpi_create_subarray_type(
            self.cartslicer.facet_size,
            self.dest,
            self.cartslicer.starts_send,
            self.cartslicer.subsizes_send,
            dtype=self.dtype,
        )

        self.src = send_rank(self.ranknd, self.grid_size, backward=not self.backward)

        self.resizedrecvsubarray = mpi_create_subarray_type(
            self.cartslicer.facet_size,
            self.src,
            self.cartslicer.starts_recv,
            self.cartslicer.subsizes_recv,
            dtype=self.dtype,
        )

    # TODO: directly maintain the array to be updated within the communicator?
    def update_borders(self, local_array):
        """Update the borders of a local array using the communication scheme
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
        return

    def remove(self):
        """Trigger object finalizer (clean-up)."""
        return self._finalizer()

    @property
    def removed(self):
        """Check whether the object has been finalized."""
        return not self._finalizer.alive
