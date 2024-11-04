import weakref

import numpy as np
from mpi4py import MPI

from mcmc.utils.communicators import BaseCommunicator


def setup_border_circular_update(
    ndims, itemsize, tile_size, overlap_size, grid_size, ranknd, backward=True
):
    r"""Source, destination types and ranks to perform the distributed adjoint
    of the circular padding operator (for `double` format data).

    Set-up destination and source data types and process ranks to perform
    communications involved in the adjoint of the circular padding operator
    within an nD Cartesian communicator.

    Parameters
    ----------
    ndims : int
        Number of dimensions of the Cartesian grid.
    itemsize : int
        Size in bytes of an item from the array to be sent.
    tile_size : numpy.ndarray[int]
        Size of the tiles (partition of the cartesian domain).
    overlap_size : numpy.ndarray[int]
        Overlap size between contiguous facets.
    grid_size : numpy.ndarray of int
        Dimension of the MPI process grid (Cartesian communicator).
    ranknd : numpy.ndarray of int
        nD rank of the current process in the Cartesian process grid.
    backward : bool, optional
        Direction of the overlap along each axis, by default True.

    Returns
    -------
    dest : list[int]
        List of process ranks to which the current process sends data.
    src : list[int]
        List of process ranks from which the current process receives data.
    border_resizedsendsubarray: list[MPI subarray]
        Custom MPI subarray to exchange data for the adjoint circular padding
        operator.
    recv_buffer : numpy.ndarray or None
        Temporary buffer to compute adjoint circular padding (data
        aggregation).

    Note
    ----
    Function appropriate only for subarrays of type float (``numpy.float64``).
    Will trigger a segfault error otherwise.
    """

    # * defining custom types to communicate non-contiguous arrays in the
    # directions considered
    border_sendsubarray = ndims * [None]
    border_resizedsendsubarray = ndims * [None]
    recv_buffer = ndims * [None]

    # ! separating diagnoals not functional yet
    # reception buffer to ensure the "junction" between the dimensions is
    # correctly accounted for
    # diag_sendsubarray = (ndims - 1) * [None]
    # diag_resizedsendsubarray = (ndims - 1) * [None]
    # diag_buffer = (ndims - 1) * [None]

    # last_rank = np.zeros(ndims - 1, dtype="i")
    # first_rank = np.zeros(ndims - 1, dtype="i")
    # for d in range(1, ndims):
    #     last_ranknd = np.r_[
    #         grid_size[: d + 1] - 1, np.zeros(ndims - (d + 1), dtype="i")
    #     ]
    #     last_rank[d - 1] = np.ravel_multi_index(last_ranknd, grid_size)
    #     # last_ranknd = np.r_[grid_size[:d+1] - 1, ranknd[d+1:]]
    #     # last_rank[d-1] = np.ravel_multi_index(last_ranknd, grid_size)

    #     # first_ranknd = np.r_[np.zeros(d+1, dtype="i"), ranknd[d+1:]]
    #     # first_rank[d-1] = np.ravel_multi_index(last_ranknd, grid_size)
    # diag_src = (ndims - 1) * [-1]
    # diag_dest = (ndims - 1) * [-1]
    # rank = np.ravel_multi_index(ranknd, grid_size)

    # * rank of processes involved in the communications
    src = ndims * [-1]  # MPI.PROC_NULL
    dest = ndims * [-1]

    # * comm. along each dimension
    if backward:
        # size of the temporary buffer before the adjoint circular padding operation
        adjoint_sizes = tile_size + (ranknd == 0) * overlap_size

        # communications along axes
        for k in range(ndims):
            if overlap_size[k] > 0:
                if ranknd[k] == grid_size[k] - 1:
                    # reception buffers
                    src[k] = np.ravel_multi_index(
                        np.r_[ranknd[:k], 0, ranknd[k + 1 :]], grid_size
                    )
                    recv_buffer[k] = np.zeros(
                        np.r_[
                            adjoint_sizes[:k], overlap_size[k], adjoint_sizes[k + 1 :]
                        ],
                        dtype="d",
                    )
                    # send (no send)

                if ranknd[k] == 0:
                    # send
                    dest[k] = np.ravel_multi_index(
                        np.r_[ranknd[:k], grid_size[k] - 1, ranknd[k + 1 :]], grid_size
                    )
                    subsizes = np.r_[
                        adjoint_sizes[:k], overlap_size[k], adjoint_sizes[k + 1 :]
                    ]
                    starts = np.zeros(ndims, dtype="i")
                    border_sendsubarray[k] = MPI.DOUBLE.Create_subarray(
                        adjoint_sizes, subsizes, starts, order=MPI.ORDER_C
                    )
                    border_resizedsendsubarray[k] = border_sendsubarray[
                        k
                    ].Create_resized(0, overlap_size[k] * itemsize)
                    border_resizedsendsubarray[k].Commit()

        # communications along "diagonal" (junction between axes)
        # ! diagonal not picking the right region for now, to be revised
        # ! check subsizes / create_resized, -> read again MPI documentation
        # for k in range(ndims - 1):
        #     # send if first buffer
        #     if rank == 0:  # first_rank[k]:
        #         diag_dest[k] = last_rank[k]
        #         subsizes = np.r_[overlap_size[: k + 2], adjoint_sizes[k + 2 :]]
        #         starts = np.zeros(ndims)
        #         diag_sendsubarray[k] = MPI.DOUBLE.Create_subarray(
        #             adjoint_sizes, subsizes, starts, order=MPI.ORDER_C
        #         )
        #         diag_resizedsendsubarray[k] = diag_sendsubarray[k].Create_resized(
        #             0, overlap_size[0] * itemsize
        #         )
        #         diag_resizedsendsubarray[k].Commit()
        #         # no reception

        #     # receive if last buffer
        #     if rank == last_rank[k]:
        #         # reception buffer
        #         diag_src[k] = 0  # first_rank[k]
        #         diag_buffer[k] = np.zeros(
        #             np.r_[overlap_size[: k + 2], adjoint_sizes[k + 2 :]],
        #             dtype="d",
        #         )
        #         # no send

    # TODO: to be updated
    else:
        adjoint_sizes = tile_size + (ranknd == grid_size - 1) * overlap_size
        for k in range(ndims):
            if overlap_size[k] > 0:
                # send and reception on the same process if grid_size[d] = 1

                # reception buffer
                if ranknd[k] == 0:
                    # reception buffer
                    src[k] = np.ravel_multi_index(
                        np.r_[ranknd[:k], grid_size[k] - 1, ranknd[k + 1 :]], grid_size
                    )
                    recv_buffer[k] = np.zeros(
                        np.r_[
                            adjoint_sizes[:k], overlap_size[k], adjoint_sizes[k + 1 :]
                        ],
                        dtype="d",
                    )
                    # send (no send)

                if ranknd[k] == grid_size[k] - 1:
                    # send
                    dest[k] = np.ravel_multi_index(
                        np.r_[ranknd[:k], 0, ranknd[k + 1 :]], grid_size
                    )
                    subsizes = np.r_[
                        adjoint_sizes[:k], overlap_size[k], adjoint_sizes[k + 1 :]
                    ]
                    starts = np.r_[
                        np.zeros(k, dtype="i"),
                        adjoint_sizes[k] - overlap_size[k],
                        np.zeros(ndims - k - 1, dtype="i"),
                    ]
                    border_sendsubarray[k] = MPI.DOUBLE.Create_subarray(
                        adjoint_sizes, subsizes, starts, order=MPI.ORDER_C
                    )
                    border_resizedsendsubarray[k] = border_sendsubarray[
                        k
                    ].Create_resized(0, overlap_size[k] * itemsize)
                    border_resizedsendsubarray[k].Commit()
                    # reception (no reception)

    return (
        dest,
        src,
        adjoint_sizes,
        border_resizedsendsubarray,
        recv_buffer,
    )


def free_custom_mpi_types(border_resizedsendsubarray, isvalid_comm):
    r"""Freeing custom MPI resized types.

    Parameters
    ----------
    border_resizedsendsubarray : list of mpi4py.MPI.Datatype, of size ``d``
        Custom MPI subarray type describing the data sent by the current
        process, as returned by ``mpi4py.MPI.Datatype.Create_subarray``.
    isvalid_comm : numpy.ndarray of bool, of size ``d``
        Boolean vector indicating wether each of the ``d`` possible
        communications are valid for the current worker (e.g., absence of
        neighbour, ...).
    """

    ndims = len(border_resizedsendsubarray)

    # free custom MPI types
    for d in range(ndims):
        if border_resizedsendsubarray[d] is not None:
            border_resizedsendsubarray[d].Free()
    # ! separating diagonals not functional yet
    # for d in range(ndims - 1):
    #     if diag_resizedsendsubarray[d] is not None:
    #         diag_resizedsendsubarray[d].Free()


class AdjointCircularPadding(BaseCommunicator):
    """Communicator for the adjoint circular padding.

    Attributes
    ----------
    comm : mpi4py.MPI.Comm
        Underlying MPI communicator.
    grid_size : list of int, of size ``d``
        Number of workers along each of the ``d`` dimensions of the
        communicator grid.
    itemsize : numpy.dtype.itemsize
        Size (in bytes) of the scalar type to be handled during the
        communications.
    tile_size : numpy.ndarray of int, of size ``d``
        Number of elements along each of the ``d`` dimensions of the tile
        handled by the current process.
    overlap_size : numpy.ndarray of int, of size ``d``
        Size of the overlap between the array handled by two different
        workers.
    backward : bool, optional
        Direction of the overlap between facets along all the axis (True
        for backward overlap, False for forward overlap). By default True.
    """

    def __init__(
        self,
        comm,
        cartcomm,
        grid_size,
        itemsize,
        tile_size,
        overlap_size,
        backward=True,
    ):
        """AdjointCircularPadding constructor.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            Underlying MPI communicator.
        cartcomm : mpi4py.MPI.Cartcomm
            Underlying Cartesian MPI communicator.
        grid_size : list of int, of size ``d``
            Number of workers along each of the ``d`` dimensions of the
            communicator grid.
        itemsize : numpy.dtype.itemsize
            Size (in bytes) of the scalar type to be handled during the
            communications.
        tile_size : numpy.ndarray of int, of size ``d``
            Number of elements along each of the ``d`` dimensions of the tile
            handled by the current process.
        overlap_size : numpy.ndarray of int, of size ``d``
            Size of the overlap between the array handled by two different
            workers.
        backward : bool, optional
            Direction of the circular padding along each axis (True
            for backward overlap, False for forward overlap). By default False.

        Raises
        ------
        ValueError
            ``overlap_size`` and ``grid_size`` must contain the same number of
            element.
        """
        super(AdjointCircularPadding, self).__init__(
            comm,
            grid_size,
            itemsize,
            tile_size,
        )
        self.overlap_size = overlap_size
        self.ranknd = np.array(cartcomm.Get_coords(self.rank), dtype="i")
        self.backward = backward
        self.tile_size = tile_size

        # configure communication scheme
        (
            self.dest,
            self.src,
            self.facet_size,
            self.border_resizedsendsubarray,
            self.recv_buffer,
        ) = setup_border_circular_update(
            self.ndims,
            self.itemsize,
            self.tile_size,
            self.overlap_size,
            self.grid_size,
            self.ranknd,
            backward=self.backward,
        )

        # slice to fold border
        self.slice_folding = self.ndims * [np.s_[:]]
        # slice to crop array after folding / aggregation
        self.slice_cropping = self.ndims * [np.s_[:]]
        # slice to add junction between consecutive axes
        # self.slice_junction = (self.ndims - 1) * [None]

        if self.backward:
            for d in range(self.ndims):
                if self.ranknd[d] == 0:
                    self.slice_cropping[d] = np.s_[self.overlap_size[d] :]

                self.slice_folding[d] = tuple(
                    d * [np.s_[:]]
                    + [np.s_[-self.overlap_size[d] :]]
                    + (self.ndims - (d + 1)) * [np.s_[:]]
                )
            # ! not functional yet if separating "diagonals" (async comms)
            # for d in range(self.ndims - 1):
            #     if self.rank == self.last_rank[d]:
            #         self.slice_junction[d] = tuple(
            #             [np.s_[-self.overlap_size[k] :] for k in range(d + 2)]
            #             + (self.ndims - d - 2) * [np.s_[:]]
            #         )
        else:
            for d in range(self.ndims):
                if self.ranknd[d] == self.grid_size[d] - 1:
                    self.slice_cropping[d] = np.s_[: -self.overlap_size[d]]

                self.slice_folding[d] = tuple(
                    d * [np.s_[:]]
                    + [np.s_[: self.overlap_size[d]]]
                    + (self.ndims - (d + 1)) * [np.s_[:]]
                )
            # ! not functional yet if separating "diagonals" (async comms)
            # for d in range(self.ndims - 1):
            #     if self.rank == self.last_rank[d]:
            #         self.slice_junction[d] = tuple(
            #             [np.s_[: self.overlap_size[d]] for k in range(d + 2)]
            #             + (self.ndims - d - 2) * [np.s_[:]]
            #         )

        self.slice_cropping = tuple(self.slice_cropping)

        # setup finalizer
        self.isvalid_comm = self.overlap_size > 0

        self._finalizer = weakref.finalize(
            self,
            free_custom_mpi_types,
            self.border_resizedsendsubarray,
            self.isvalid_comm,
        )

    def update_borders(self, local_array):
        """Adjoint circular padding.

        Update the borders of a local array using the communication scheme
        defined in the communicator. Aggregates contributions from the border
        of the global array distributed across processes. A separate buffer is
        used for this operation.

        Parameters
        ----------
        local_array : numpy.ndarray, with ``d`` dimensions, float entries
            Local array to be updated through communications.

        Note
        ----
        - The input array, ``local_array`` is updated in-place.
        - The function will trigger a segfault error if the type of
          ``local_array`` is different from ``float`` (due to the hard-coded
          MPI type used to defined ``self.resizedsendsubarray``)
        - MPI aggregation can be performed in-place using MIP windows, see
        ``MPI_Win_allocate`` (see ``mpi4py.MPI.Win.Accumulate``).
        """
        for d in range(self.ndims):
            requests = []
            if self.backward:
                if self.overlap_size[d] > 0:
                    if self.ranknd[d] == 0:
                        requests.append(
                            self.comm.Isend(
                                [local_array, 1, self.border_resizedsendsubarray[d]],
                                self.dest[d],
                            )
                        )

                    if self.ranknd[d] == self.grid_size[d] - 1:
                        requests.append(
                            self.comm.Irecv(
                                self.recv_buffer[d],
                                source=self.src[d],
                            )
                        )
            else:
                if self.overlap_size[d] > 0:
                    if self.ranknd[d] == 0:
                        requests.append(
                            self.comm.Irecv(
                                self.recv_buffer[d],
                                source=self.src[d],
                            )
                        )

                    if self.ranknd[d] == self.grid_size[d] - 1:
                        requests.append(
                            self.comm.Isend(
                                [local_array, 1, self.border_resizedsendsubarray[d]],
                                self.dest[d],
                            )
                        )

            if len(requests) > 0:
                MPI.Request.Waitall(requests)

            # aggregating result before moving to the next dimension
            if self.recv_buffer[d] is not None:
                local_array[self.slice_folding[d]] += self.recv_buffer[d]

        # cropping
        local_array = local_array[self.slice_cropping]

        return local_array

    # def fold(self, local_array):
    #     """Adjoint circular padding. Aggregates contributions from the border
    #     of the global array distributed across processes.

    #     Parameters
    #     ----------
    #     local_array : numpy.ndarray, with ``d`` dimensions, float entries
    #         Local array to be updated through communications.
    #     """
    #     for d in range(0, self.ndims):
    #         if self.recv_buffer[d] is not None:
    #             local_array[self.slice_folding[d]] += self.recv_buffer[d]

    #     # ! diag_buffer[d] not picking the right terms at the moment, to be fixed (influenced by src / dest)
    #     print(
    #         "Process {}: diag_src={}, diag_dest={}, self.diag_buffer={}".format(
    #             self.rank, self.diag_src, self.diag_dest, self.diag_buffer
    #         )
    #     )

    #     for d in range(0, self.ndims - 1):
    #         if self.diag_buffer[d] is not None:
    #             local_array[self.slice_junction[d]] += self.diag_buffer[d]

    #     # cropping
    #     local_array = local_array[self.slice_cropping]

    #     return local_array

    def remove(self):
        """Trigger object finalizer (clean-up)."""
        return self._finalizer()

    @property
    def removed(self):
        """Check whether the object has been finalized."""
        return not self._finalizer.alive
