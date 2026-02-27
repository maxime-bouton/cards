"""Implementation of a generic Cartesian slicer object to support array
communications on a Cartesian grid of processes."""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)

import numpy as np

from mcmc.slicer.cartesian_tesselation import local_split_range_nd
from mcmc.slicer.comm_slicer import CommSlicer


def compute_local_buffer_size(
    ranknd: np.ndarray,
    grid_size: np.ndarray,
    tile_size: np.ndarray,
    overlap_size: np.ndarray,
    backward: bool = True,
):
    """Compute the size of the chunk of convolution hold by the current worker.

    Parameters
    ----------
    ranknd : numpy.ndarray[int]
        Multi-dimensional rank of the current process in the nD Cartesian grid of MPI
        processes.
    grid_size : numpy.ndarray[int]
        Number of processes along each dimension of the nD Cartesian grid.
    tile_size : numpy.ndarray[int]
        Size of the non-overlapping image tile underlying the overlapping facets.
    overlap_size : numpy.ndarray[int]
        Size of the overlap along each dimension.
    backward : bool, optional
        Direction of the overlap in the Cartesian grid along all the dimensions (forward
        or backward overlap), by default True.

    Returns
    -------
    facet_size : numpy.ndarray[int]
        Size of the overlapping facet handled by the current process.
    """

    not_edge = (ranknd > 0) if backward else (ranknd < grid_size - 1)
    condition = not_edge & (grid_size > 1)

    facet_size = tile_size + condition * overlap_size
    return facet_size


def create_slice_buffers(
    ranknd: np.ndarray,
    grid_size: np.ndarray,
    tile_size: np.ndarray,
    overlap_size: np.ndarray,
    backward: bool = True,
):
    """Create a slice to extract tile (elements exclusively handled by the current
    process) from the associated facet (gathering both the tile and the elements
    received by the current process from neighbour processes).

    Parameters
    ----------
    ranknd : numpy.array[int]
        Multi-dimensional rank of the current process in the nD Cartesian grid of MPI
        processes.
    grid_size : numpy.array[int]
        Size of the MPI Cartesian grid considered.
    tile_size : numpy.array[int]
        Shape of the tile array.
    overlap_size : numpy.array[int]
        Amount of overlap along each axis between two consecutive processes.
    backward : bool, optional
        Direction of the overlap between two consecutive processes along all axes, by
        default True.

    Returns
    -------
    slice
        Slice object to extract the tile from the corresponding facet buffer.
    """

    if backward:  # select the last elements of the facet buffer
        return tuple([np.s_[-tile_size[d] :] for d in range(ranknd.size)])
    else:  # select the first elements of the facet buffer
        return tuple([np.s_[: tile_size[d]] for d in range(ranknd.size)])


def create_slice_sync_send_recv(
    ranknd, grid_size, facet_size, send_size, recv_size, backward=True
):
    """Slice to extract received and sent elements from a worker (sync).

    Create slice to extract elements received or sent by a worker from the facet buffer
    during the synchronous communication phases occuring along each axis of a Cartesian
    grid of processes.

    Parameters
    ----------
    ranknd : numpy.ndarray[int]
        Multi-dimensional rank of the current process in the nD Cartesian grid of MPI
        processes.
    grid_size : numpy.ndarray[int]
        Number of processes along each dimension of the nD Cartesian grid.
    facet_size : numpy.ndarray[int]
        Size of the overlapping facet.
    send_size : numpy.ndarray[int]
        Size of overlap along each axis to determine arrays to be sent.
    recv_size : numpy.ndarray[int]
        Size of overlap along each axis to determine arrays to be received.
    backward : bool, optional
        Direction of the overlap in the Cartesian grid along all the
        dimensions (forward or backward overlap), by default True.

    Returns
    -------
    slice_send : tuple[slice]
        Slice to extract the message sent from the current process within the
        local facet.
    slice_recv : tuple[slice]
        Slice to extract the message received by the current process into the
        local facet.
    starts_send : np.array[int]
        Starting index of each subarray to be sent to other workers from current facet.
    subsizes_send : np.array[int]
        Size of each subarray to be sent to other workers from current facet.
    starts_recv : np.array[int]
        Starting index of each subarray received, inserted into current facet.
    subsizes_recv : np.array[int]
        Size of each subarray received, to inserted into current facet.
    """
    ndims = ranknd.size

    slice_send = ndims * [None]
    slice_recv = ndims * [None]

    starts_send = np.zeros((ndims, ndims), dtype="i")
    subsizes_send = np.zeros((ndims, ndims), dtype="i")
    starts_recv = np.zeros((ndims, ndims), dtype="i")
    subsizes_recv = np.zeros((ndims, ndims), dtype="i")

    if backward:
        s_size = np.zeros(ndims, dtype="i")
        r_size = facet_size.copy()
        for d in range(ndims):
            s_size[d] = facet_size[d] - send_size[d]
            slice_send[d] = tuple([np.s_[s_size[q] :] for q in range(ndims)])
            starts_send[d] = s_size
            subsizes_send[d] = facet_size - s_size
            s_size[d] = 0

            r_size[d] = recv_size[d]
            slice_recv[d] = tuple([np.s_[: r_size[q]] for q in range(ndims)])
            starts_recv[d] = 0
            subsizes_recv[d] = r_size
            r_size[d] = facet_size[d]
    else:
        s_size = facet_size.copy()
        r_size = np.zeros(ndims, dtype="i")
        for d in range(ndims):
            s_size[d] = send_size[d]
            slice_send[d] = tuple([np.s_[: s_size[q]] for q in range(ndims)])
            starts_send[d] = 0
            subsizes_send[d] = s_size
            s_size[d] = facet_size[d]

            r_size[d] = facet_size[d] - recv_size[d]
            slice_recv[d] = tuple([np.s_[r_size[q] :] for q in range(ndims)])
            starts_recv[d] = r_size
            subsizes_recv[d] = facet_size - r_size
            r_size[d] = 0

    return (
        slice_send,
        slice_recv,
        starts_send,
        subsizes_send,
        starts_recv,
        subsizes_recv,
    )


def create_slice_async_send_recv(
    ranknd, grid_size, facet_size, send_size, recv_size, backward=True
):
    """Slice to extract received and sent elements from a worker (async).

    Create slice to extract elements received or sent by a worker from the
    facet buffer during the asynchronous communication phases occuring along the different combination of axes in a Cartesian grid of processes.

    Parameters
    ----------
    ranknd : numpy.ndarray[int]
        Multi-dimensional rank of the current process in the nD Cartesian grid
        of MPI processes.
    grid_size : numpy.ndarray[int]
        Number of processes along each dimension of the nD Cartesian grid.
    facet_size : numpy.ndarray[int]
        Size of the overlapping overlapping facet.
    send_size : numpy.ndarray[int]
        Size of overlap along each axis to determine arrays to be sent.
    recv_size : numpy.ndarray[int]
        Size of overlap along each axis to determine arrays to be received.
    backward : bool, optional
        Direction of the overlap in the Cartesian grid along all the
        dimensions (forward or backward overlap), by default True.

    Returns
    -------
    slice_send : tuple[slice]
        Slice to extract the message sent from the current process within the
        local facet.
    slice_recv : tuple[slice]
        Slice to extract the message received by the current process into the
        local facet.
    starts_send : np.array[int]
        Starting index of each subarray to be sent to other workers from current facet.
    subsizes_send : np.array[int]
        Size of each subarray to be sent to other workers from current facet.
    starts_recv : np.array[int]
        Starting index of each subarray received, inserted into current facet.
    subsizes_recv : np.array[int]
        Size of each subarray received, to inserted into current facet.
    """
    ndims = ranknd.size
    ncomms = ndims * (ndims - 1) + 1

    slice_send = ncomms * [None]
    slice_recv = ncomms * [None]
    starts_send = np.zeros((ncomms, ndims), dtype="i")
    subsizes_send = np.zeros((ncomms, ndims), dtype="i")
    starts_recv = np.zeros((ncomms, ndims), dtype="i")
    subsizes_recv = np.zeros((ncomms, ndims), dtype="i")

    mask = np.full(ndims, False)

    if backward:
        size = facet_size - send_size
        r_size = recv_size.copy()

        for k in range(ndims - 1):
            mask[k] = True
            for d in range(ndims):
                r_size[mask] = size[mask]
                slice_send[k * ndims + d] = tuple(
                    [np.s_[r_size[q] :] for q in range(ndims)]
                )
                starts_send[k * ndims + d] = r_size
                subsizes_send[k * ndims + d] = facet_size - r_size

                slice_recv[k * ndims + d] = ndims * [np.s_[:]]
                for q in range(ndims):
                    if mask[q]:
                        slice_recv[k * ndims + d][q] = np.s_[: recv_size[q]]
                        starts_recv[k * ndims + d, q] = 0
                        subsizes_recv[k * ndims + d, q] = recv_size[q]
                    else:
                        slice_recv[k * ndims + d][q] = np.s_[recv_size[q] :]
                        starts_recv[k * ndims + d, q] = recv_size[q]
                        subsizes_recv[k * ndims + d, q] = facet_size[q] - recv_size[q]
                slice_recv[k * ndims + d] = tuple(slice_recv[k * ndims + d])

                r_size[mask] = recv_size[mask]
                mask = np.roll(mask, 1)

        slice_send[-1] = tuple([np.s_[size[d] :] for d in range(ndims)])
        starts_send[-1] = size
        subsizes_send[-1] = facet_size - size

        slice_recv[-1] = tuple([np.s_[: recv_size[d]] for d in range(ndims)])
        starts_recv[-1] = 0
        subsizes_recv[-1] = recv_size

    else:  # forward overlap
        size = facet_size - recv_size
        s_size = size.copy()

        for k in range(ndims - 1):
            mask[k] = True
            for d in range(ndims):
                s_size[mask] = send_size[mask]

                slice_send[k * ndims + d] = tuple(
                    [np.s_[: s_size[q]] for q in range(ndims)]
                )
                starts_send[k * ndims + d] = 0
                subsizes_send[k * ndims + d] = s_size

                slice_recv[k * ndims + d] = ndims * [np.s_[:]]
                for q in range(ndims):
                    if mask[q]:
                        slice_recv[k * ndims + d][q] = np.s_[size[q] :]
                        starts_recv[k * ndims + d, q] = size[q]
                        subsizes_recv[k * ndims + d, q] = facet_size[q] - size[q]
                    else:
                        slice_recv[k * ndims + d][q] = np.s_[: size[q]]
                        starts_recv[k * ndims + d, q] = 0
                        subsizes_recv[k * ndims + d, q] = size[q]
                slice_recv[k * ndims + d] = tuple(slice_recv[k * ndims + d])

                s_size[mask] = size[mask]
                mask = np.roll(mask, 1)

        slice_send[-1] = tuple([np.s_[: send_size[d]] for d in range(ndims)])
        starts_send[-1] = 0
        subsizes_send[-1] = send_size

        slice_recv[-1] = tuple([np.s_[size[d] :] for d in range(ndims)])
        starts_recv[-1] = size
        subsizes_recv[-1] = facet_size - size

    return (
        slice_send,
        slice_recv,
        starts_send,
        subsizes_send,
        starts_recv,
        subsizes_recv,
    )


class CartesianCommSlicer(CommSlicer):
    """Slicer object underlying Cartesian communications.

    Attributes
    ----------
    ranknd : numpy.ndarray[int]
        n-dimensional rank of the current process, i.e., rank along each
        axis of the Cartesian MPI process grid considered.
    grid_size : numpy.ndarray[int]
        Number of processes along each axis of the Cartesian MPI process
        grid considered.
    global_buffer_size : numpy.ndarray[int]
        Size of the global array split across the Cartesian grid of MPI
        processes considered.
    send_size : numpy.ndarray[int]
        Dimensions of the subarray to be sent across each axis of the
        process grid.
    recv_size : numpy.ndarray[int]
        Dimensions of the subarray to be received across each axis of the
        process grid.
    backward : bool, optional
        Direction of the overlap (opposite of the direction of communications), by default True.
    ndims : int
        Number of dimensions ("axes") of the arrays handled.
    tile_range : np.array[int, 2]
        Start and end index delimiting the portion of the global array handled by the current process.
    tile_size : numpy.ndarray[int]
        Size of the tile handled by the current process.
    facet_size : numpy.ndarray[int]
        Size of the facet handled by the current process.
    slice_send : tuple[slice]
        Slice used to extract subarrays to be sent to other workers from the
        facet handled by the current process, assuming successive communications along each dimensions of the array.
    slice_recv : tuple[slice]
        Slice used to extract subarrays received from other workers from the
        facet handled by the current process, assuming successive communications along each dimensions of the array.
    slice_async_send : tuple[slice]
        Slice used to extract subarrays to be sent to other workers from the
        facet handled by the current process, considering the communication pattern required by asynchronous communications.
    slice_async_recv : tuple[slice]
        Slice used to extract subarrays received from other workers from the
        facet handled by the current process, considering the communication pattern required by asynchronous communications.
    slice_facet_to_tile : tuple[slice]
        Slice used to extract tile from the local facet handled by the current
        process.
    starts_send : np.array[int]
        Starting index of each subarray to be sent to other workers from current facet (synchronous communications).
    starts_recv : np.array[int]
        Starting index of each subarray received, inserted into current facet (synchronous communications).
    starts_async_send : np.array[int]
        Starting index of each subarray to be sent to other workers from current facet (asynchronous communications).
    starts_async_recv : np.array[int]
        Starting index of each subarray received, inserted into current facet (asynchronous communications).
    subsizes_send : np.array[int]
        Size of each subarray to be sent to other workers from current facet (synchronous communications).
    subsizes_recv : np.array[int]
        Size of each subarray received, to inserted into current facet (synchronous communications).
    subsizes_async_send : np.array[int]
        Size of each subarray to be sent to other workers from current facet (asynchronous communications).
    subsizes_async_recv : np.array[int]
        Size of each subarray received, to inserted into current facet (asynchronous communications).

    Note
    ----
    The word `tile` refers to the portion of the global array exclusively
    handled by the current process. The word `facet` refers to the local array
    handled by the process, i.e., including elements sent by neighbouring
    workers to the current process ("overlap").
    """

    # FIXME: clarify wording: tile (exclusive ownership), facet: all data (with overlap)

    def __init__(
        self,
        ranknd,
        grid_size,
        global_buffer_size,
        send_size,
        recv_size,
        backward=True,
        tile_range=None,
    ):
        """Generate slices to manipulate sub-arrays involved in the Cartesian decomposition of a global array.

        Slicer object to extract tile from facet and global array, and access
        subarrays sent or recevied from the current MPI process in a Cartesian
        grid.

        Parameters
        ----------
        ranknd : numpy.ndarray[int]
            n-dimensional rank of the current process, i.e., rank along each
            axis of the Cartesian MPI process grid considered.
        grid_size : numpy.ndarray[int]
            Number of processes along each axis of the Cartesian MPI process
            grid considered.
        global_buffer_size : numpy.ndarray[int]
            Size of the global array split across the Cartesian grid of MPI
            processes considered.
        send_size : numpy.ndarray[int]
            Dimensions of the subarray to be sent across each axis of the
            process grid.
        recv_size : numpy.ndarray[int]
            Dimensions of the subarray to be received across each axis of the
            process grid.
        backward : bool, optional
            Direction of the overlap (opposite of the direction of communications), by default True.
        tile_range : numpy.ndarray[int] or None, optional
            Index of the elements from the global array exclusively handled by the current process, defining a subarray. By default None, so that it is directly specified by the object itself, dividing the global array evenly across the different workers.

        Raises
        ------
        ValueError
            All entries in send_size should be positive.
        ValueError
            All entries in recv_size should be positive.
        ValueError
            All entries in tile_size should be greater than send_size.
        """
        self.ranknd = ranknd

        # number of axes of the problem
        self.ndims = self.ranknd.size

        if np.any(send_size < 0):
            raise ValueError(r"All entries in send_size should be positive.")
        if np.any(recv_size < 0):
            raise ValueError(r"All entries in recv_size should be positive.")

        # number of points communicated along each axis (send / reception)
        # ! set overlap size to 0 for axes not split across different MPI
        # ! processes (rank boundary or limitation grid_size)
        # ! assuming there is no cyclic communication along any axis
        self.send_size = send_size.copy()
        self.recv_size = recv_size.copy()
        if backward:
            self.send_size[(grid_size <= 1) | (ranknd == grid_size - 1)] = 0
            self.recv_size[(grid_size <= 1) | (ranknd == 0)] = 0
        else:
            self.send_size[(grid_size <= 1) | (ranknd == 0)] = 0
            self.recv_size[(grid_size <= 1) | (ranknd == grid_size - 1)] = 0

        # direction of the reception comm between consecutive processes in the
        # Cartesian grid
        self.backward = backward

        # elements owned by the process (i.e., whose update is exclusively
        # handled by the current process)
        if tile_range is None:
            self.tile_range = local_split_range_nd(
                grid_size,
                global_buffer_size,
                self.ranknd,
            )
        else:
            self.tile_range = tile_range
        self.tile_size = (self.tile_range[:, 1] - self.tile_range[:, 0] + 1).astype(int)

        # check overlap size is smaller than tile size (more general communication topology required otherwise
        if np.any(self.send_size > self.tile_size):
            raise ValueError(
                r"All entries in tile_size should be greater than send_size."
            )

        # ! see if issue here due to redefinition of recv_size
        # number of elements handled (i.e., locally owned + received)
        self.facet_size = compute_local_buffer_size(
            self.ranknd,
            grid_size,
            self.tile_size,
            self.recv_size,
            backward=self.backward,
        )

        # extracting sent/received information from facet buffer
        # slicing for sync. communications (ndims communication phases)
        (
            self.slice_send,
            self.slice_recv,
            self.starts_send,
            self.subsizes_send,
            self.starts_recv,
            self.subsizes_recv,
        ) = create_slice_sync_send_recv(
            self.ranknd,
            grid_size,
            self.facet_size,
            self.send_size,
            self.recv_size,
            backward=self.backward,
        )
        # slicing for async. communications
        (
            self.slice_async_send,
            self.slice_async_recv,
            self.starts_async_send,
            self.subsizes_async_send,
            self.starts_async_recv,
            self.subsizes_async_recv,
        ) = create_slice_async_send_recv(
            self.ranknd,
            grid_size,
            self.facet_size,
            self.send_size,
            self.recv_size,
            backward=self.backward,
        )

        # extracting tile from facet
        self.slice_facet_to_tile = create_slice_buffers(
            self.ranknd,
            grid_size,
            self.tile_size,
            self.recv_size,
            backward=self.backward,
        )

        super(CartesianCommSlicer, self).__init__(
            grid_size,
            global_buffer_size,
        )

    def _get_slice_global_buffer_to_tile(self):
        """Create slice to insert tile into, or extract tile from, the full
        array."""
        # global_buffer[slice_local_range] = local_tile
        return tuple(
            np.s_[int(self.tile_range[d, 0]) : int(self.tile_range[d, 1]) + 1]
            for d in range(self.ndims)
        )
