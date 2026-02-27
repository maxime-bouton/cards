"""Utility functions to creating MPI subarray datatypes."""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)

import mpi4py.util.dtlib as mpilib
import numpy as np
from mpi4py import MPI


def get_ranknd(rank, grid_size):
    """Generate the nD rank of a process from its linear rank.

    Generate the nD rank of a process from its linear rank within a Cartesian grid of processes of shape ``grid_size``.

    Parameters
    ----------
    rank : int
        Linear rank of a process.
    grid_size : numpy.ndarray[int]
        Shape of the Cartesian grid of processes.

    Returns
    -------
    numpy.ndarray[int]
        nD rank of the process.
    """
    return np.array(np.unravel_index(rank, grid_size), dtype="i")


def mpi_create_subarray_type(
    array_size,
    comm_rank,
    comm_starts,
    comm_subsizes,
    dtype=np.float64,
):
    r"""Source, destination types and ranks to update facet borders (for
    `double` format data).

    Set-up destination and source data types and process ranks to communicate
    facet borders within an nD Cartesian communicator. Diagonal communications
    (involving more than a single dimension) are not separated from the other
    communications.

    Parameters
    ----------
    array_size : numpy.ndarray[int]
        Size of the array from which a subarray needs to be extracted.
    comm_size : numpy.ndarray[int]
        Size of the communication ("overlap size") with contiguous workers
        along each dimension.
    comm_rank : numpy.ndarray[int]
        List of process ranks with which the current process needs to communicate. Contains the value ``MPI_PROC_NULL`` for any invalid communication.
    comm_starts : numpy.ndarray[int]
        nD index of the starting point of the subarray to be extracted.
    comm_subsizes : numpy.ndarray[int]
        Shape of the subarray to be extracted.
    dtype : numpy.dtype, optional
        Type of the buffer over which the communicator is defined (required
        to define sub-arrays), by default np.float64.

    Returns
    -------
    resizedsubarray : list[MPI subarray]
        Custom MPI subarray type describing the subarray to be communicated to another process (see `mpi4py.MPI.Datatype.Create_subarray <https://mpi4py.github.io/usrman/reference/mpi4py.MPI.Datatype.html?highlight=create%20subarray#mpi4py.MPI.Datatype.Create_subarray>`_).

    Note
    ----
    For synchronous communications, ``ndims_comm`` communications maximum need
    to be performed per worker (one along each axis of the Cartesian grid.
    Whenever a communication along an axis is not valid, ``comm_rank`` is
    expected to contain the value ``MPI.PROC_NULL`` in the corresponding entry.
    """
    # Useful references
    # 1. https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node83.htm#Node83
    # 2. https://events.prace-ri.eu/event/1049/sessions/3350/attachments/1330/2362/Advanced%20MPI-%20User-defined%20datatypes.pdf
    # Size := number of bytes that have to be transferred -> favor the wording "shape"
    # Extent := spans from first to last byte (including all holes).
    # True extent := spans from first to last true byte (excluding holes at begin+end)
    # Automatic holes at the end for necessary alignment purpose
    # Additional holes at begin and by lb and ub markers: MPI_TYPE_CREATE_RESIZED
    # Basic datatypes: Size = Extent = number of bytes used by the compiler

    # number of communications to be performed
    ncomms = comm_rank.size

    if comm_rank.size == 0:
        resizedsubarray = []
    else:
        mpi_datatype = mpilib.from_numpy_dtype(dtype)

        # size in bytes of an item from the array to be sent.
        # itemsize = np.dtype(dtype).itemsize

        # defining custom types to communicate non-contiguous arrays
        resizedsubarray = ncomms * [None]

        for comm_id in range(ncomms):
            if (
                comm_rank[comm_id] > MPI.PROC_NULL
            ):  # if valid communication, create new Datatype, keep None otherwise
                subarray = mpi_datatype.Create_subarray(
                    array_size,
                    comm_subsizes[comm_id],
                    comm_starts[comm_id],
                    order=MPI.ORDER_C,
                )

                [lb, extent] = subarray.Get_extent()
                resizedsubarray[comm_id] = subarray.Create_resized(lb, extent)
                resizedsubarray[comm_id].Commit()

    return resizedsubarray


def free_custom_mpi_types(
    resizedsendsubarray: list[MPI.Datatype], resizedrecvsubarray: list[MPI.Datatype]
):
    r"""Freeing custom MPI types.

    Parameters
    ----------
    resizedsendsubarray : list of MPI.Datatype, of size ``d``
        Custom MPI subarray type describing the data sent by the current
        process, as returned by ``MPI.Datatype.Create_subarray``.
    resizedrecvsubarray : list of MPI.Datatype, of size ``d``
        Custom MPI subarray type describing the data received by the current
        process, as returned by ``MPI.Datatype.Create_subarray``.
    """
    ncomms = len(resizedsendsubarray)

    for comm_id in range(ncomms):
        if resizedsendsubarray[comm_id] is not None:
            resizedsendsubarray[comm_id].Free()
        if resizedrecvsubarray[comm_id] is not None:
            resizedrecvsubarray[comm_id].Free()
    pass
