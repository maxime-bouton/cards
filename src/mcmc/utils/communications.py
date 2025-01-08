"""Core set of functions implementing the communications leveraged from the
communicators defined in :mod:`dsgs.utils.communicators`.
"""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# reference: P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.

from collections import deque

import numpy as np
from mpi4py import MPI

# TODO: investigate
# [persistent communications](https://mpi4py.readthedocs.io/en/stable/overview.
# html#persistent-communications)


def split_range(nchunks, N, overlap=0, backward=True, circular=False):
    r"""Tessellates :math:`\{ 0, \dotsc , N-1 \}` into multiple subsets.

    Tessellates :math:`\{ 0, \dotsc , N-1 \}` into (non-)overlapping
    subsets, each containing approximately the same number of indices.

    Parameters
    ----------
    nchunks : int
        Number of segments.
    N : int
        Total number of segments.
    overlap : int, optional
        Defines overlap size between segments (if any). Defaults to 0.
    backward : bool, optional
        Direction of the overlap, if any (backward or forward). Defaults to
        True.
    circular : bool, optional
        Allows circular overlap, if any (backward or forward). Defaults to
        False.

    Returns
    -------
    numpy.ndarray[int]
        Start and end index of each segment. Shape: ``(nchunks, 2)``.

    Raises
    ------
    ValueError
        Error if the overlap is greater than the size of a segment.
    """

    splits = np.linspace(-1, N - 1, num=nchunks + 1, dtype="i")

    if overlap > np.floor(N / nchunks):
        raise ValueError(r"More than 100% overlap between two consecutive segments")

    if overlap <= 0:
        # w/o overlap
        rg = np.concatenate((splits[:-1][:, None] + 1, splits[1:][:, None]), axis=1)
    else:
        # with overlap
        if backward:
            # overlap towards the left (backward)
            rg = np.concatenate(
                (
                    np.array((splits[0] + 1, *(splits[1:-1] + 1 - overlap)))[:, None],
                    splits[1:][:, None],
                ),
                axis=1,
            )
        else:
            # overlap towards the right (forward)
            rg = np.concatenate(
                (
                    splits[:-1][:, None] + 1,
                    np.array((*(splits[1:-1] + overlap), splits[-1]))[:, None],
                ),
                axis=1,
            )
    return rg


def local_split_range(nchunks, N, index, overlap=0, backward=True):
    r"""Return the portion of :math:`\{ 0, \dotsc , N-1 \}` handled by a
    process.

    Return the portion of :math:`\{ 0, \dotsc , N-1 \}`, tessellated into
    (non-)overlapping subsets, owned by a process, with nchunks processes in
    total.

    Parameters
    ----------
    nchunks : int
        Total number of segments.
    N : int
        Total number of indices.
    index : int
        Rank of the current process.
    overlap : int, optional
        Overlap size between consecutive segments (if any), by default 0.
    backward : bool, optional
        Direction of the overlap, if any (backward or forward), by default
        True.

    Returns
    -------
    numpy.ndarray[int]
        Start and end index of the segment: shape ``(2,)``.

    Raises
    ------
    ValueError
        Error if ``index`` is greater than ``nchunks-1``.
    ValueError
        Error if the overlap is greater than the size of a segment.
    """

    if nchunks <= index:
        raise ValueError(
            r"Index should be taken in [0, ..., nchunks-1], with nchunks={0}".format(
                nchunks
            )
        )
    step = N / nchunks
    if overlap > np.floor(step):
        raise ValueError(r"More than 100% overlap between two consecutive segments")

    start = -1 + index * step
    stop = np.rint(start + step)
    start = np.rint(start)
    rg = np.array([start + 1, stop], dtype="i")
    # if the facet overlaps with a neighbour
    if overlap > 0:
        if backward:
            # overlap towards the left (backward)
            if index > 0:
                rg[0] -= overlap
        else:
            # overlap towards the right (forward)
            if index < nchunks - 1:
                rg[-1] += overlap
    return rg


def local_split_range_nd(nchunks, N, index, overlap=None, backward=True):
    r"""Return the portion of :math:`\{ 0, \dotsc , N-1 \}` (nD range
    of indices) handled by a process.

    Return the portion of :math:`\{ 0, \dotsc , N-1 \}`, tessellated
    into (non-)overlapping subsets along each dimension, owned by a process.

    Parameters
    ----------
    nchunks : numpy.ndarray[int]
        Total number of segments along each dimension.
    N : numpy.ndarray[int]
        Total number of indices along each dimension.
    index : numpy.ndarray[int]
        Rank of the current process along each dimension.
    overlap : numpy.ndarray[int], optional
        Overlap size between consecutive segments along each dimension, by
        default None.
    backward : bool, optional
        Direction of the overlap (forward or backward), by default True.

    Returns
    -------
    numpy.ndarray[int]
        Start and end index of the nD segment along each dimension:
        shape ``(ndims, 2)``.

    Raises
    ------
    ValueError
        Error if any index is greater than ``nchunks-1``.
    ValueError
        Error if any overlap size is greater than the size of the corresponding
        segment.
    """

    # TODO: see how to make it compatible with a 1D implementation
    if np.any(nchunks <= index):
        raise ValueError(
            r"Index should be taken in [0, ..., nchunks-1], with nchunks={0}".format(
                nchunks
            )
        )
    step = N / nchunks
    if overlap is not None:
        if np.any(overlap > np.floor(step)):
            raise ValueError(r"More than 100% overlap between two consecutive segments")
    start = -1 + index * step
    stop = (start + step).astype(np.int64)
    start = start.astype(np.int64)
    rg = np.concatenate(((start + 1)[:, None], stop[:, None]), axis=1)

    if overlap is not None:
        if backward:
            rg[np.logical_and(index > 0, overlap > 0), 0] = (
                rg[np.logical_and(index > 0, overlap > 0), 0]
                - overlap[np.logical_and(index > 0, overlap > 0)]
            )
        else:
            rg[np.logical_and(index < nchunks - 1, overlap > 0), 1] = (
                rg[np.logical_and(index < nchunks - 1, overlap > 0), 1]
                + overlap[np.logical_and(index < nchunks - 1, overlap > 0)]
            )
    return np.squeeze(rg)


# ! WARNING: may result in empty tiles, thus changing the total number of
# ! facets
def rebalance_tile_nd(
    tile_indices, ranknd, N, grid_size, overlap_size
):  # pragma: no cover
    r""" "Resize the local tiles to rebalance data sizes across the grid.

    Resize the local tiles across the grid to compensate for the increase in
    the data size for border facets.

    Parameters
    ----------
    tile_indices : numpy.ndarray[int]
        Indices of the pixels from the full array composing the tile (first
        point, last point). Shape: ``(ndims, 2)``.
    ranknd : numpy.ndarray[int]
        Rank of the current process (multiple dimensions).
    N : numpy.ndarray[int]
        Total number of indices along each dimension.
    grid_size : numpy.ndarray[int]
        Size of the process grid.
    overlap_size : numpy.ndarray[int]
        Overlap size between consecutive facets across the grid along each
        dimension.

    Returns
    -------
    numpy.ndarray[int]
        Updated indices defining the local tile: shape ``(ndims, 2)``.
    """

    # safeguard: trigger rebalance only if the data chunk associated with the
    # last facet is expected to be twice as large as a tile
    ndims = ranknd.size
    ideal_size = np.floor(N / grid_size + overlap_size)
    tile_size = tile_indices[:, 1] - tile_indices[:, 0] + 1

    for d in range(ndims):
        if (
            overlap_size[d] > 0
            and grid_size[d] > 1
            and ideal_size[d] > 2 * tile_size[d]
        ):
            if overlap_size[d] < grid_size[d]:
                offset = 1
                # increase size of the first overlap_size[d] tiles along dim. d
                if ranknd[d] < overlap_size[d] - 1:
                    tile_indices[d, 0] += ranknd[d] * offset
                    tile_indices[d, 1] += (ranknd[d] + 1) * offset
            else:
                # increase size of the first grid_size[d]-1 along dim. d
                offset = np.floor(overlap_size[d] / grid_size[d])
                if ranknd[d] < grid_size[d] - 1:
                    tile_indices[d, 0] += ranknd[d] * offset
                    tile_indices[d, 1] += (ranknd[d] + 1) * offset
                else:
                    tile_indices[d, 0] += (
                        ranknd[d] * offset
                    )  # ! only modify starting point (last facet)

    return tile_indices


# ! NOT USED at the moment
def local_split_range_symmetric_nd(
    nchunks, N, index, overlap_pre=np.array([0]), overlap_post=np.array([0])
):
    r"""Return the portion of :math:`\{ 0, \dotsc , N-1 \}` (nD range of
    indices) handled by a process (symmetric version).

    Return the portion of :math:`\{ 0, \dotsc , N-1 \}`, tessellated into
    (non-overlapping subsets along each dimension, owned by a process.

    Parameters
    ----------
    nchunks : numpy.ndarray[int]
        Number of nd-segments along each dimension.
    N : numpy.ndarray[int]
        Total number of indices along each dimension.
    index : numpy.ndarray[int]
        Index of the process.
    overlap_pre : numpy.ndarray[int], optional
        Defines overlap with preceding segment, by default ``np.array([0])``.
    overlap_post : numpy.ndarray[int], optional
        Defines overlap with following segment, by default ``np.array([0])``.

    Returns
    -------
    numpy.ndarray[int]
        Start and end index of the nD segment along each dimension. Shape:
        ``(ndims, 2)``.

    Raises
    ------
    ValueError
        Error if any index is greater than ``nchunks-1``.
    ValueError
        Error if any overlap is greater than the size of a segment.
    """

    # TODO: see how to make it compatible with a 1D implementation
    if np.any(nchunks <= index):
        raise ValueError(
            r"Index should be taken in [0, ..., nchunks-1], with nchunks={0}".format(
                nchunks
            )
        )
    step = N / nchunks
    if np.any(overlap_pre > np.floor(step)) or np.any(overlap_post > np.floor(step)):
        raise ValueError(r"More than 100% overlap between two consecutive segments")
    start = -1 + index * step
    stop = (start + step).astype(np.int64)
    start = start.astype(np.int64)
    rg = np.concatenate(((start + 1)[:, None], stop[:, None]), axis=1)
    rg[np.logical_and(index > 0, overlap_pre > 0), 0] = (
        rg[np.logical_and(index > 0, overlap_pre > 0), 0]
        - overlap_pre[np.logical_and(index > 0, overlap_pre > 0)]
    )

    rg[np.logical_and(index < nchunks - 1, overlap_post > 0), 1] = (
        rg[np.logical_and(index < nchunks - 1, overlap_post > 0), 1]
        + overlap_post[np.logical_and(index < nchunks - 1, overlap_post > 0)]
    )
    return np.squeeze(rg)


def split_range_interleaved(nchunks, N):
    r"""Tessellates :math:`\{ 0, \dotsc , N-1 \}` into interleaved subsets.

    Tessellates :math:`\{ 0, \dotsc , N-1 \}` into subsets of interleaved
    indices, each containing approximately the same number of indices
    (downsampling of :math:`\{ 0, \dotsc , N-1 \}`).

    Parameters
    ----------
    nchunks : int
        Total number of segments.
    N : int
        Total number of indices.

    Returns
    -------
    list[slice]
        List of slices to extract the indices corresponding to each set.

    Raises
    ------
    ValueError
        Error if the overlap is greater than the size of a segment.
    """

    if nchunks > N:
        raise ValueError(
            r"Number of segments nchunks={0} greater than the dimension N={1}".format(
                nchunks, N
            )
        )

    return [np.s_[k:N:nchunks] for k in range(nchunks)]


def local_split_range_interleaved(nchunks, N, index):
    r"""Tessellates :math:`\{ 0, \dotsc , N-1 \}` into interleaved
    subsets.

    Tessellates :math:`\{ 0, \dotsc , N-1 \}` into subsets of
    interleaved indices, each containing approximately the same number of
    indices (downsampling of :math:`\{ 0, \dotsc , N-1 \}`).

    Parameters
    ----------
    nchunks : int
        Total number of segments.
    N : int
        Total number of indices.
    index : int
        Index identifying the chunk considered.

    Returns
    -------
    slice
        Slice to extract the indices corresponding to the corresponding
        segment.

    Raises
    ------
    ValueError
        Error if the index is greater than ``nchunks-1``.
    ValueError
        Error if the overlap is greater than the size of a segment.
    """

    if nchunks <= index:
        raise ValueError(
            r"Index should be taken in [0, ..., nchunks-1], with nchunks={0}".format(
                nchunks
            )
        )
    if nchunks > N:
        raise ValueError(
            r"Number of segments nchunks={0} greater than the dimension N={1}".format(
                nchunks, N
            )
        )

    return np.s_[index:N:nchunks]


def get_neighbour(ranknd, grid_size, disp):
    """1D rank of a neighbour of the current MPI process.

    Returns the 1D rank of the neighbour of the current MPI process,
    corresponding to a pre-defined displacement vector `disp` in the nD
    Cartesian grid.

    Parameters
    ----------
    ranknd : numpy.ndarray[int]
        nD rank of the current process
    grid_size : numpy.ndarray[int]
        Size of the Cartesian process grid (number of processes along each
        dimension)
    disp : numpy.ndarray[int]
        Displacement vector to obtain the rank of a neighbour process.

    Returns
    -------
    int
        1D rank of the neighbour process
    """

    return np.ravel_multi_index(ranknd + disp, grid_size)


# TODO: move elsewhere, specific to convolution
def slice_valid_coefficients(ranknd, grid_size, overlap_size):
    r"""Helper elements to extract the valid local convolution coefficients.

    Returns slice to select the valid local convolution coefficients, with the
    necessary padding parameters to implement the adjoint (zero-padding)
    operator.

    Parameters
    ----------
    ranknd : numpy.ndarray[int]
        Rank of the process in a Cartesian nD grid of MPI processes.
    grid_size : numpy.ndarray[int]
        Size of the MPI process grid.
    overlap_size : numpy.ndarray[int]
        Overlap between contiguous facets along each dimension.

    Returns
    -------
    valid_coefficients : tuple[slice]
        Slice to extract valid coefficients from the local convolutions.

    Raises
    ------
    AssertionError
        `ranknd`, `grid_size` and `overlap_size` must all have the save shape.
    """

    ndims = ranknd.size

    if not (grid_size.size == ndims and overlap_size.size == ndims):
        raise AssertionError(
            r"`ranknd`, `grid_size` and `overlap_size` must have the save \
                shape"
        )

    L = ndims * [None]
    R = ndims * [None]

    for d in range(ndims):
        if grid_size[d] > 1 and overlap_size[d] > 0:
            if ranknd[d] > 0 and ranknd[d] < grid_size[d] - 1:
                L[d] = overlap_size[d]
                R[d] = -overlap_size[d]
            elif ranknd[d] == grid_size[d] - 1:
                L[d] = overlap_size[d]
                R[d] = None
            else:
                L[d] = 0
                R[d] = -overlap_size[d]
        else:
            L[d] = 0
            R[d] = None

    valid_coefficients = tuple([np.s_[L[d] : R[d]] for d in range(ndims)])

    return valid_coefficients


def get_local_slice(ranknd, grid_size, overlap_size, backward=True):
    r"""Slice to extract the pixels specific to a given worker.

    Get the slice corresponding to the elements exclusively handled by the
    current process (i.e., remove the overlap from overlapping facets).

    Parameters
    ----------
    ranknd : numpy.ndarray[int]
        Rank of the current process in the nD Cartesian grid of MPI processes.
    grid_size : numpy.ndarray[int]
        Size of the process grid
    overlap_size : numpy.ndarray[int]
        Size of the overlap between contiguous facets.
    backward : bool, optional
        Orientation of the overlap along the dimensions, by default True.

    Returns
    -------
    tuple[slice]
        Slice to extract the coefficients specifically handled by the current
        process.

    Raises
    ------
    AssertionError
        `ranknd`, `grid_size` and `overlap_size` must all have the save shape.
    """

    ndims = ranknd.size

    if not (grid_size.size == ndims and overlap_size.size == ndims):
        raise AssertionError(
            r"`ranknd`, `grid_size` and `overlap_size` must have the save \
                shape"
        )

    local_slice = ndims * [np.s_[:]]
    isvalid_splitting = np.logical_and(grid_size > 1, overlap_size > 0)

    if backward:
        for d in range(ndims):
            if ranknd[d] > 0 and isvalid_splitting[d]:
                local_slice[d] = np.s_[overlap_size[d] :]
    else:
        for d in range(ndims):
            if ranknd[d] < grid_size[d] - 1 and isvalid_splitting[d]:
                local_slice[d] = np.s_[: -overlap_size[d]]

    return tuple(local_slice)


# ! NOT USED
def get_local_slice_border(ranknd, grid_size, overlap_size, border_size, backward=True):
    r"""Slice to extract the pixels specific to a given worker.

    Get the slice corresponding to the elements exclusively handled by the
    current process (i.e., remove the overlap from overlapping facets).

    Parameters
    ----------
    ranknd : numpy.ndarray[int]
        Rank of the current process in the nD Cartesian grid of MPI processes.
    grid_size : numpy.ndarray[int]
        Size of the process grid
    overlap_size : numpy.ndarray[int]
        Size of the overlap between contiguous facets.
    border_size : numpy.ndarray[int]
        Size of the border added around the overall image (boundary extension).
    backward : bool, optional
        Orientation of the overlap along the dimensions, by default True.

    Returns
    -------
    tuple[slice]
        Slice to extract the coefficients specifically handled by the current
        process.

    Raises
    ------
    AssertionError
        `ranknd`, `grid_size` and `overlap_size` must all have the save shape.
    """

    ndims = ranknd.size

    if not (grid_size.size == ndims and overlap_size.size == ndims):
        raise AssertionError(
            r"`ranknd`, `grid_size` and `overlap_size` must have the save \
                shape"
        )

    local_slice = ndims * [np.s_[:]]
    isvalid_splitting = np.logical_and(grid_size > 1, border_size > 0)

    # ! need to adjust offest due to boundary extension
    if backward:
        for d in range(ndims):
            if isvalid_splitting[d]:
                if ranknd[d] == 0:
                    local_slice[d] = np.s_[border_size[d] :]
                elif ranknd[d] < grid_size[d] - 1:
                    local_slice[d] = np.s_[overlap_size[d] :]
                else:  # ranknd[d] == grid_size[d] - 1
                    local_slice[d] = np.s_[overlap_size[d] : -border_size[d]]
    else:
        for d in range(ndims):
            if isvalid_splitting[d]:
                if ranknd[d] == grid_size[d] - 1:
                    local_slice[d] = np.s_[: -border_size[d]]
                elif ranknd[d] > 0:
                    local_slice[d] = np.s_[: -overlap_size[d]]
                else:  # ranknd[d] == 0
                    local_slice[d] = np.s_[border_size[d] : -overlap_size[d]]

    return tuple(local_slice)


# def get_border_slice(ranknd, grid_size, overlap_size, backward=True):
#     r"""Slice to extract the border pixels on a given worker.

#     Get the slice corresponding to the elements on the borders of the current process (i.e., retain only the overlap area from the facets).

#     Parameters
#     ----------
#     ranknd : numpy.ndarray[int]
#         Rank of the current process in the nD Cartesian grid of MPI processes.
#     grid_size : numpy.ndarray[int]
#         Size of the process grid
#     overlap_size : numpy.ndarray[int]
#         Size of the overlap between contiguous facets.
#     backward : bool, optional
#         Orientation of the overlap along the dimensions, by default True.

#     Returns
#     -------
#     tuple[slice]
#         Slice to extract the coefficients overlapping with neighbour processes.

#     Raises
#     ------
#     AssertionError
#         `ranknd`, `grid_size` and `overlap_size` must all have the save shape.
#     """

#     ndims = ranknd.size

#     if not (grid_size.size == ndims and overlap_size.size == ndims):
#         raise AssertionError(
#             r"`ranknd`, `grid_size` and `overlap_size` must have the save \
#                 shape"
#         )

#     border_slice = ndims * [np.s_[:]]
#     isvalid_splitting = np.logical_and(grid_size > 1, overlap_size > 0)

#     if backward:
#         for d in range(ndims):
#             if ranknd[d] > 0 and isvalid_splitting[d]:
#                 border_slice[d] = np.s_[: overlap_size[d]]
#     else:
#         for d in range(ndims):
#             if ranknd[d] < grid_size[d] - 1 and isvalid_splitting[d]:
#                 border_slice[d] = np.s_[-overlap_size[d] :]

#     return tuple(border_slice)


# ! USED ONLY IN TESTS, not in actual code implementation
def isvalid_communication(ranknd, grid_size, overlap_size, N, backward=True):
    r"""Check which of the possible communications, along each axis or
    combination of axes ("diagonal"), are valid.

    This function checks which communications (including "diagonal"
    communications, i.e., along a combination of multiple axes) are valid, and
    returns the rank the source (`src`) process, destination process (`dest`),
    and the number of pixels.

    Parameters
    ----------
    ranknd : numpy.ndarray[int]
        Rank of the current process in the nD Cartesian grid of MPI processes.
    grid_size : numpy.ndarray[int]
        Size of the process grid.
    overlap_size : numpy.ndarray[int]
        Size of the overlap between contiguous facets.
    N : numpy.ndarray[int]
        Total number of pixels along each direction.
    backward : bool, optional
        Orientation of the overlap along the dimensions, by default True.

    Returns
    -------
    dest : numpy.ndarray[int]
        Rank of the destination process (-1 if not valid).
    src : numpy.ndarray[int]
        Rank of the source process (-1 if not valid).
    isvalid_dest : array[bool]
        List of valid communications (send).
    isvalid_src : array[bool]
        List of valid communications (receive).
    sizes_dest : numpy.ndarray[int]
        Number of pixels (along each axis) to be sent for each communication.
        Shape: ``(ndims * (ndims - 1) + 1, ndims)``.
    sizes_src : numpy.ndarray[int]
        Number of pixels (along each axis) to be received for each
        communication.
        Shape: ``(ndims * (ndims - 1) + 1, ndims)``..
    start_src : numpy.ndarray[int]
        Coordinates of the starting point needed to extract pixels to be
        communicated. Shape: ``(ndims * (ndims - 1) + 1, ndims)``.
    """

    # Example: order of communications in 3D, [x, y, z]
    # [x, y, z, xy, yz, zx, xyz]

    # TODO: add definition of the subtypes to send / receive the necessary data
    # start, size, types, ...
    # TODO: add more comments

    ndims = ranknd.size
    isvalid_splitting = np.logical_and(overlap_size > 0, grid_size > 1)

    if backward:
        isvalid_rank_dest = ranknd < grid_size - 1
        isvalid_rank_src = ranknd > 0
        dest_value = 1
    else:
        isvalid_rank_dest = ranknd > 0
        isvalid_rank_src = ranknd < grid_size - 1
        dest_value = -1

    isvalid_dest = np.full(ndims * (ndims - 1) + 1, False, dtype="bool")
    isvalid_src = np.full(ndims * (ndims - 1) + 1, False, dtype="bool")
    src = (ndims * (ndims - 1) + 1) * [MPI.PROC_NULL]
    dest = (ndims * (ndims - 1) + 1) * [MPI.PROC_NULL]
    sizes_dest = np.zeros((ndims * (ndims - 1) + 1, ndims), dtype="i")
    sizes_src = np.zeros((ndims * (ndims - 1) + 1, ndims), dtype="i")
    start_src = np.zeros((ndims * (ndims - 1) + 1, ndims), dtype="i")
    disp = np.zeros(ndims, dtype="i")

    overlap_size[np.logical_not(isvalid_rank_src)] = 0
    Ns = N - overlap_size

    for k in range(1, ndims):
        sel_id = deque(k * [True] + (ndims - k) * [False])
        nsel_id = deque(k * [False] + (ndims - k) * [True])
        for d in range(ndims):
            c0 = np.all(isvalid_splitting[sel_id])
            s = (k - 1) * ndims + d

            if c0 and np.all(isvalid_rank_dest[sel_id]):
                isvalid_dest[s] = True
                sizes_dest[s, sel_id] = overlap_size[sel_id]
                sizes_dest[s, nsel_id] = Ns[nsel_id]
                disp[sel_id] = dest_value
                dest[s] = get_neighbour(ranknd, grid_size, disp)
            if c0 and np.all(isvalid_rank_src[sel_id]):
                isvalid_src[s] = True
                sizes_src[s, sel_id] = overlap_size[sel_id]
                sizes_src[s, nsel_id] = Ns[nsel_id]
                disp[sel_id] = -dest_value
                src[s] = get_neighbour(ranknd, grid_size, disp)
                start_src[s, np.logical_not(sel_id)] = overlap_size[
                    np.logical_not(sel_id)
                ]

            disp[sel_id] = 0
            sel_id.rotate(1)
            nsel_id.rotate(1)

    # all dimensions active
    isvalid_src[-1] = np.all(isvalid_splitting) and np.all(isvalid_rank_src)
    isvalid_dest[-1] = np.all(isvalid_splitting) and np.all(isvalid_rank_dest)
    if isvalid_src[-1]:
        src[-1] = get_neighbour(
            ranknd, grid_size, np.full(ndims, -dest_value, dtype="i")
        )
        sizes_src[-1, :] = overlap_size

    if isvalid_dest[-1]:
        dest[-1] = get_neighbour(
            ranknd, grid_size, np.full(ndims, dest_value, dtype="i")
        )
        sizes_dest[-1, :] = overlap_size

    return (
        dest,
        src,
        isvalid_dest,
        isvalid_src,
        sizes_dest,
        sizes_src,
        start_src,
    )


def setup_border_update(
    cartcomm, ndims, itemsize, facet_size, overlap_size, backward=True
):
    r"""Source, destination types and ranks to update facet borders (for
    `double` format data).

    Set-up destination and source data types and process ranks to communicate
    facet borders within an nD Cartesian communicator. Diagonal communications
    (involving more than a single dimension) are not separated from the other
    communications.

    Parameters
    ----------
    cartcomm : mpi4py.MPI.Cartcomm
        Cartesian topology intracommunicator.
    ndims : int
        Number of dimensions of the Cartesian grid.
    itemsize : int
        Size in bytes of an item from the array to be sent.
    facet_size : numpy.ndarray[int]
        Size of the overlapping facets.
    overlap_size : numpy.ndarray[int]
        Overlap size between contiguous facets.
    backward : bool, optional
        Direction of the overlap along each axis, by default True.

    Returns
    -------
    dest : list[int]
        List of process ranks to which the current process sends data.
    src : list[int]
        List of process ranks from which the current process receives data.
    resizedsendsubarray : list[MPI subarray]
        Custom MPI subarray type describing the data array sent by the current
        process (see `mpi4py.MPI.Datatype.Create_subarray <https://mpi4py.github.io/usrman/reference/mpi4py.MPI.Datatype.html?highlight=create%20subarray#mpi4py.MPI.Datatype.Create_subarray>`_).
    resizedrecvsubarray : list[MPI subarray]
        Custom MPI subarray type describing the data array sent by the current
        process (see `mpi4py.MPI.Datatype.Create_subarray <https://mpi4py.github.io/usrman/reference/mpi4py.MPI.Datatype.html?highlight=create%20subarray#mpi4py.MPI.Datatype.Create_subarray>`_).

    Note
    ----
    Function appropriate only for MPI subarrays of type float
    (``numpy.float64``). Will trigger a segfault error otherwise.
    """

    # * defining custom types to communicate non-contiguous arrays in the
    # directions considered
    sendsubarray = []
    recvsubarray = []
    resizedsendsubarray = []
    resizedrecvsubarray = []

    sizes = facet_size  # size of local array
    sM = sizes - overlap_size

    # * rank of processes involved in the communications
    src = ndims * [MPI.PROC_NULL]
    dest = ndims * [MPI.PROC_NULL]

    # * comm. along each dimension
    if backward:
        for k in range(ndims):
            if overlap_size[k] > 0:
                # ! if there is no overlap along a dimension, make sure there
                # ! is no communication
                [src[k], dest[k]] = cartcomm.Shift(k, 1)

                # send buffer
                subsizes = np.r_[sizes[:k], overlap_size[k], sizes[k + 1 :]]
                starts = np.r_[
                    np.zeros(k, dtype="i"),
                    sM[k],
                    np.zeros(ndims - k - 1, dtype="i"),
                ]
                sendsubarray.append(
                    MPI.DOUBLE.Create_subarray(
                        sizes, subsizes, starts, order=MPI.ORDER_C
                    )
                )
                resizedsendsubarray.append(
                    sendsubarray[-1].Create_resized(0, overlap_size[k] * itemsize)
                )  # ! see if extent is still fine for more than 2 dimensions
                # [lb, extent] = sendsubarray[-1].Get_extent()
                # resizedsendsubarray.append(sendsubarray[-1].Create_resized(lb, extent))
                resizedsendsubarray[-1].Commit()

                # recv buffer
                subsizes = np.r_[sizes[:k], overlap_size[k], sizes[k + 1 :]]
                starts = np.zeros(ndims, dtype="i")
                recvsubarray.append(
                    MPI.DOUBLE.Create_subarray(
                        sizes, subsizes, starts, order=MPI.ORDER_C
                    )
                )
                resizedrecvsubarray.append(
                    recvsubarray[-1].Create_resized(0, overlap_size[k] * itemsize)
                )
                # [lb, extent] = recvsubarray[-1].Get_extent()
                # resizedrecvsubarray.append(recvsubarray[-1].Create_resized(lb, extent))
                resizedrecvsubarray[-1].Commit()
            else:
                resizedsendsubarray.append(None)
                resizedrecvsubarray.append(None)
    else:
        for k in range(ndims):
            if overlap_size[k] > 0:
                # ! if there is no overlap along a dimension, make sure there
                # ! is no communication
                [src[k], dest[k]] = cartcomm.Shift(k, -1)

                # recv buffer
                subsizes = np.r_[sizes[:k], overlap_size[k], sizes[k + 1 :]]
                starts = np.r_[
                    np.zeros(k, dtype="i"),
                    sM[k],
                    np.zeros(ndims - k - 1, dtype="i"),
                ]
                recvsubarray.append(
                    MPI.DOUBLE.Create_subarray(
                        sizes, subsizes, starts, order=MPI.ORDER_C
                    )
                )
                resizedrecvsubarray.append(
                    recvsubarray[-1].Create_resized(0, overlap_size[k] * itemsize)
                )  # ! see if extent is still fine for more than 2 dimensions
                # [lb, extent] = recvsubarray[-1].Get_extent()
                # resizedrecvsubarray.append(recvsubarray[-1].Create_resized(lb, extent))
                resizedrecvsubarray[-1].Commit()

                # send buffer
                subsizes = np.r_[sizes[:k], overlap_size[k], sizes[k + 1 :]]
                starts = np.zeros(ndims, dtype="i")
                sendsubarray.append(
                    MPI.DOUBLE.Create_subarray(
                        sizes, subsizes, starts, order=MPI.ORDER_C
                    )
                )
                resizedsendsubarray.append(
                    sendsubarray[-1].Create_resized(0, overlap_size[k] * itemsize)
                )
                # [lb, extent] = sendsubarray[-1].Get_extent()
                # resizedsendsubarray.append(sendsubarray[-1].Create_resized(lb, extent))

                resizedsendsubarray[-1].Commit()
            else:
                resizedsendsubarray.append(None)
                resizedrecvsubarray.append(None)

    return dest, src, resizedsendsubarray, resizedrecvsubarray


# ! USED IN TESTS ONLY, not in actual code implementation
def setup_border_circular_update(
    cartcomm, ndims, itemsize, facet_size, overlap_size, ranknd, backward=True
):
    r"""Source, destination types and ranks to update facet borders (for
    `double` format data).

    Set-up destination and source data types and process ranks to communicate
    facet borders within an nD Cartesian communicator. Diagonal communications
    (involving more than a single dimension) are not separated from the other
    communications.

    Parameters
    ----------
    cartcomm : mpi4py.MPI.Cartcomm
        Cartesian topology intracommunicator.
    ndims : int
        Number of dimensions of the Cartesian grid.
    itemsize : int
        Size in bytes of an item from the array to be sent.
    facet_size : numpy.ndarray[int]
        Size of the overlapping facets.
    overlap_size : numpy.ndarray[int]
        Overlap size between contiguous facets.
    backward : bool, optional
        Direction of the overlap along each axis, by default True.

    Returns
    -------
    dest : list[int]
        List of process ranks to which the current process sends data.
    src : list[int]
        List of process ranks from which the current process receives data.
    resizedsendsubarray : list[MPI subarray]
        Custom MPI subarray type describing the data array sent by the current
        process (see `mpi4py.MPI.Datatype.Create_subarray <https://mpi4py.github.io/usrman/reference/mpi4py.MPI.Datatype.html?highlight=create%20subarray#mpi4py.MPI.Datatype.Create_subarray>`_).
    resizedrecvsubarray : list[MPI subarray]
        Custom MPI subarray type describing the data array sent by the current
        process (see `mpi4py.MPI.Datatype.Create_subarray <https://mpi4py.github.io/usrman/reference/mpi4py.MPI.Datatype.html?highlight=create%20subarray#mpi4py.MPI.Datatype.Create_subarray>`_).

    Note
    ----
    Function appropriate only for subarrays of type float (``numpy.float64``).
    Will trigger a segfault error otherwise.
    """

    # * defining custom types to communicate non-contiguous arrays in the
    # directions considered
    sendsubarray = []
    recvsubarray = []
    resizedsendsubarray = []
    resizedrecvsubarray = []

    border_sendsubarray = []
    border_recvsubarray = []
    border_resizedsendsubarray = []
    border_resizedrecvsubarray = []

    sizes = facet_size  # size of local array
    sM = sizes - overlap_size

    # * rank of processes involved in the communications
    src = ndims * [MPI.PROC_NULL]
    dest = ndims * [MPI.PROC_NULL]

    # * comm. along each dimension
    # TODO: backward option to be adapted to the setting considered
    if backward:
        for k in range(ndims):
            if overlap_size[k] > 0:
                # ! if there is no overlap along a dimension, make sure there
                # ! is no communication
                [src[k], dest[k]] = cartcomm.Shift(k, 1)

                # send buffer
                subsizes = np.r_[sizes[:k], overlap_size[k], sizes[k + 1 :]]
                # ! change send area for first worker along the dimension (will
                # ! communicate with the last worker for border exchange)
                if ranknd[k] == cartcomm.dims[k] - 1:
                    starts = np.r_[
                        np.zeros(k, dtype="i"),
                        sM[k] - overlap_size[k],
                        np.zeros(ndims - k - 1, dtype="i"),
                    ]
                else:
                    starts = np.r_[
                        np.zeros(k, dtype="i"),
                        sM[k],
                        np.zeros(ndims - k - 1, dtype="i"),
                    ]
                sendsubarray.append(
                    MPI.DOUBLE.Create_subarray(
                        sizes, subsizes, starts, order=MPI.ORDER_C
                    )
                )
                resizedsendsubarray.append(
                    sendsubarray[-1].Create_resized(0, overlap_size[k] * itemsize)
                )  # ! see if extent is still fine for more than 2 dimensions
                resizedsendsubarray[-1].Commit()

                # recv buffer
                subsizes = np.r_[sizes[:k], overlap_size[k], sizes[k + 1 :]]
                starts = np.zeros(ndims, dtype="i")
                recvsubarray.append(
                    MPI.DOUBLE.Create_subarray(
                        sizes, subsizes, starts, order=MPI.ORDER_C
                    )
                )
                resizedrecvsubarray.append(
                    recvsubarray[-1].Create_resized(0, overlap_size[k] * itemsize)
                )
                resizedrecvsubarray[-1].Commit()

                # add one more communication between first and last worker
                # (along each dimension) for border exchange (last receives
                # from last)
                if ranknd[k] == cartcomm.dims[k] - 1:
                    # recv
                    subsizes = np.r_[sizes[:k], overlap_size[k], sizes[k + 1 :]]
                    starts = np.r_[
                        np.zeros(k, dtype="i"),
                        sM[k],
                        np.zeros(ndims - k - 1, dtype="i"),
                    ]
                    border_recvsubarray.append(
                        MPI.DOUBLE.Create_subarray(
                            sizes, subsizes, starts, order=MPI.ORDER_C
                        )
                    )
                    border_resizedrecvsubarray.append(
                        border_recvsubarray[-1].Create_resized(
                            0, overlap_size[k] * itemsize
                        )
                    )
                    border_resizedrecvsubarray[-1].Commit()

                    # send
                    border_resizedsendsubarray.append(None)

                elif ranknd[k] == 0:
                    # recv
                    border_resizedrecvsubarray.append(None)

                    # send
                    subsizes = np.r_[sizes[:k], overlap_size[k], sizes[k + 1 :]]
                    starts = np.r_[
                        np.zeros(k, dtype="i"),
                        overlap_size[k],
                        np.zeros(ndims - k - 1, dtype="i"),
                    ]
                    border_sendsubarray.append(
                        MPI.DOUBLE.Create_subarray(
                            sizes, subsizes, starts, order=MPI.ORDER_C
                        )
                    )
                    border_resizedsendsubarray.append(
                        border_sendsubarray[-1].Create_resized(
                            0, overlap_size[k] * itemsize
                        )
                    )
                    border_resizedsendsubarray[-1].Commit()
            else:
                resizedsendsubarray.append(None)
                resizedrecvsubarray.append(None)
    else:
        for k in range(ndims):
            if overlap_size[k] > 0:
                # ! if there is no overlap along a dimension, make sure there
                # ! is no communication
                [src[k], dest[k]] = cartcomm.Shift(k, -1)

                # recv buffer
                subsizes = np.r_[sizes[:k], overlap_size[k], sizes[k + 1 :]]
                starts = np.r_[
                    np.zeros(k, dtype="i"),
                    sM[k],
                    np.zeros(ndims - k - 1, dtype="i"),
                ]
                recvsubarray.append(
                    MPI.DOUBLE.Create_subarray(
                        sizes, subsizes, starts, order=MPI.ORDER_C
                    )
                )
                resizedrecvsubarray.append(
                    recvsubarray[-1].Create_resized(0, overlap_size[k] * itemsize)
                )  # ! see if extent is still fine for more than 2 dimensions
                resizedrecvsubarray[-1].Commit()

                # send buffer
                subsizes = np.r_[sizes[:k], overlap_size[k], sizes[k + 1 :]]
                # ! change send area for first worker along the dimension (will
                # ! communicate with the last worker for border exchange)
                if ranknd[k] == 0:
                    starts = np.r_[
                        np.zeros(k, dtype="i"),
                        overlap_size[k],
                        np.zeros(ndims - k - 1, dtype="i"),
                    ]
                else:
                    starts = np.zeros(ndims, dtype="i")
                sendsubarray.append(
                    MPI.DOUBLE.Create_subarray(
                        sizes, subsizes, starts, order=MPI.ORDER_C
                    )
                )
                resizedsendsubarray.append(
                    sendsubarray[-1].Create_resized(0, overlap_size[k] * itemsize)
                )
                resizedsendsubarray[-1].Commit()

                # add one more communication between first and last worker
                # (along each dimension) for border exchange (first receives
                # from last)
                if ranknd[k] == 0:
                    # recv
                    subsizes = np.r_[sizes[:k], overlap_size[k], sizes[k + 1 :]]
                    starts = np.zeros(ndims, dtype="i")
                    border_recvsubarray.append(
                        MPI.DOUBLE.Create_subarray(
                            sizes, subsizes, starts, order=MPI.ORDER_C
                        )
                    )
                    border_resizedrecvsubarray.append(
                        border_recvsubarray[-1].Create_resized(
                            0, overlap_size[k] * itemsize
                        )
                    )
                    border_resizedrecvsubarray[-1].Commit()

                    # send
                    border_resizedsendsubarray.append(None)

                elif ranknd[k] == cartcomm.dims[k] - 1:
                    # recv
                    border_resizedrecvsubarray.append(None)

                    # send
                    subsizes = np.r_[sizes[:k], overlap_size[k], sizes[k + 1 :]]
                    starts = np.r_[
                        np.zeros(k, dtype="i"),
                        sM[k] - overlap_size[k],
                        np.zeros(ndims - k - 1, dtype="i"),
                    ]
                    border_sendsubarray.append(
                        MPI.DOUBLE.Create_subarray(
                            sizes, subsizes, starts, order=MPI.ORDER_C
                        )
                    )
                    border_resizedsendsubarray.append(
                        border_sendsubarray[-1].Create_resized(
                            0, overlap_size[k] * itemsize
                        )
                    )
                    border_resizedsendsubarray[-1].Commit()
                else:
                    # find a way not to receive sth !!
                    border_resizedsendsubarray.append(None)
                    border_resizedrecvsubarray.append(None)

            else:
                resizedsendsubarray.append(None)
                resizedrecvsubarray.append(None)
                border_resizedsendsubarray.append(None)
                border_resizedrecvsubarray.append(None)

    return (
        dest,
        src,
        resizedsendsubarray,
        resizedrecvsubarray,
        border_resizedsendsubarray,
        border_resizedrecvsubarray,
    )


# ! USED IN SANDBOX ONLY
def setup_border_update_int(
    cartcomm, ndims, itemsize, facet_size, overlap_size, backward=True
):  # pragma: no cover
    r"""Source and destination types and ranks to update facet borders (for
    `int` format data). Only serving as a relevant MPI-based example.

    Set-up destination and source data types and process ranks to communicate
    facet borders within an nD Cartesian communicator. Diagonal communications
    (involving more than a single dimension) are not separated from the other
    communications.

    Parameters
    ----------
    cartcomm : mpi4py.MPI.Cartcomm
        Cartesian topology intracommunicator.
    ndims : int
        Number of dimensions of the Cartesian grid.
    itemsize : int
        Size in bytes of an item from the array to be sent.
    facet_size : numpy.ndarray[int]
        Size of the overlapping facets.
    overlap_size : numpy.ndarray[int]
        Overlap size between contiguous facets.
    backward : bool, optional
        Direction of the overlap along each axis, by default True.

    Returns
    -------
    dest : list[int]
        List of process ranks to which the current process sends data.
    src : list[int]
        List of process ranks from which the current process receives data.
    resizedsendsubarray : list[MPI subarray]
        Custom MPI subarray type describing the data array sent by the current
        process (see `mpi4py.MPI.Datatype.Create_subarray <https://mpi4py.github.io/usrman/reference/mpi4py.MPI.Datatype.html?highlight=create%20subarray#mpi4py.MPI.Datatype.Create_subarray>`_).
    resizedrecvsubarray : list[MPI subarray]
        Custom MPI subarray type describing the data array sent by the current
        process (see `mpi4py.MPI.Datatype.Create_subarray <https://mpi4py.github.io/usrman/reference/mpi4py.MPI.Datatype.html?highlight=create%20subarray#mpi4py.MPI.Datatype.Create_subarray>`_).

    Note
    ----
    Function appropriate only for subarrays of type int (``numpy.int32``).
    Will trigger a segfault error otherwise.
    """

    # * defining custom types to communicate non-contiguous arrays
    sendsubarray = []
    recvsubarray = []
    resizedsendsubarray = []
    resizedrecvsubarray = []

    sizes = facet_size  # size of local array
    sM = sizes - overlap_size

    # * rank of processes involved in the communications
    src = ndims * [MPI.PROC_NULL]
    dest = ndims * [MPI.PROC_NULL]

    # * comm. along each dimension
    if backward:
        for k in range(ndims):
            if overlap_size[k] > 0:
                # ! if there is no overlap along a dimension, make sure there is no
                # ! communication
                [src[k], dest[k]] = cartcomm.Shift(k, 1)

                # send buffer
                subsizes = np.r_[sizes[:k], overlap_size[k], sizes[k + 1 :]]
                starts = np.r_[
                    np.zeros(k, dtype="i"),
                    sM[k],
                    np.zeros(ndims - k - 1, dtype="i"),
                ]
                sendsubarray.append(
                    MPI.INT.Create_subarray(sizes, subsizes, starts, order=MPI.ORDER_C)
                )
                resizedsendsubarray.append(
                    sendsubarray[-1].Create_resized(0, overlap_size[k] * itemsize)
                )  # ! see if extent is still fine for more than 2 dimensions
                resizedsendsubarray[-1].Commit()

                # recv buffer
                subsizes = np.r_[sizes[:k], overlap_size[k], sizes[k + 1 :]]
                starts = np.zeros(ndims, dtype="i")
                recvsubarray.append(
                    MPI.INT.Create_subarray(sizes, subsizes, starts, order=MPI.ORDER_C)
                )
                resizedrecvsubarray.append(
                    recvsubarray[-1].Create_resized(0, overlap_size[k] * itemsize)
                )
                resizedrecvsubarray[-1].Commit()
            else:
                resizedsendsubarray.append(None)
                resizedrecvsubarray.append(None)
    else:
        for k in range(ndims):
            if overlap_size[k] > 0:
                # ! if there is no overlap along a dimension, make sure there
                # ! is no communication
                [src[k], dest[k]] = cartcomm.Shift(k, -1)

                # recv buffer
                subsizes = np.r_[sizes[:k], overlap_size[k], sizes[k + 1 :]]
                starts = np.r_[
                    np.zeros(k, dtype="i"),
                    sM[k],
                    np.zeros(ndims - k - 1, dtype="i"),
                ]
                recvsubarray.append(
                    MPI.INT.Create_subarray(sizes, subsizes, starts, order=MPI.ORDER_C)
                )
                resizedrecvsubarray.append(
                    recvsubarray[-1].Create_resized(0, overlap_size[k] * itemsize)
                )  # ! see if extent is still fine for more than 2 dimensions
                resizedrecvsubarray[-1].Commit()

                # send buffer
                subsizes = np.r_[sizes[:k], overlap_size[k], sizes[k + 1 :]]
                starts = np.zeros(ndims, dtype="i")
                sendsubarray.append(
                    MPI.INT.Create_subarray(sizes, subsizes, starts, order=MPI.ORDER_C)
                )
                resizedsendsubarray.append(
                    sendsubarray[-1].Create_resized(0, overlap_size[k] * itemsize)
                )
                resizedsendsubarray[-1].Commit()
            else:
                resizedsendsubarray.append(None)
                resizedrecvsubarray.append(None)

    return dest, src, resizedsendsubarray, resizedrecvsubarray


def setup_border_update_tv(
    cartcomm, ndims, itemsize, facet_size, overlap_size, backward=True
):
    r"""Source, destination types and ranks to update facet borders for the
    distributed gradient operator (for `double` format data).

    Set-up destination and source data types and process ranks to communicate
    facet borders within an nD Cartesian communicator. Diagonal communications
    (involving more than a single dimension) are not separated from the other
    communications.

    Parameters
    ----------
    cartcomm : mpi4py.MPI.Cartcomm
        Cartesian topology intracommunicator.
    ndims : int
        Number of dimensions of the Cartesian grid.
    itemsize : int
        Size in bytes of an item from the array to be sent.
    facet_size : numpy.ndarray[int]
        Size of the overlapping facets.
    overlap_size : numpy.ndarray[int]
        Overlap size between contiguous facets.
    backward : bool, optional
        Direction of the overlap along each axis, by default True.

    Returns
    -------
    dest : list[int]
        List of process ranks to which the current process sends data.
    src : list[int]
        List of process ranks from which the current process receives data.
    resizedsendsubarray : list[MPI subarray]
        Custom MPI subarray type describing the data array sent by the current
        process (see `mpi4py.MPI.Datatype.Create_subarray <https://mpi4py.github.io/usrman/reference/mpi4py.MPI.Datatype.html?highlight=create%20subarray#mpi4py.MPI.Datatype.Create_subarray>`_).
    resizedrecvsubarray : list[MPI subarray]
        Custom MPI subarray type describing the data array sent by the current
        process (see `mpi4py.MPI.Datatype.Create_subarray <https://mpi4py.github.io/usrman/reference/mpi4py.MPI.Datatype.html?highlight=create%20subarray#mpi4py.MPI.Datatype.Create_subarray>`_).
    """
    # TODO
    # ! try to generalize and merge with setup_border_update
    # ! idea: some dimensions are not affected by the distribution, need to
    # ! specify which are those
    # ! some elements are hard-coded for now

    # * defining custom types to communicate non-contiguous arrays in the
    # directions considered
    sendsubarray = []
    recvsubarray = []
    resizedsendsubarray = []
    resizedrecvsubarray = []

    sizes = np.empty(ndims + 1, dtype="i")
    sizes[0] = 2
    sizes[1:] = facet_size  # size of local array
    sM = facet_size - overlap_size

    # * rank of processes involved in the communications
    src = ndims * [MPI.PROC_NULL]
    dest = ndims * [MPI.PROC_NULL]

    # * comm. along each dimension
    if backward:
        for k in range(ndims):
            if overlap_size[k] > 0:
                # ! if there is no overlap along a dimension, make sure there
                # ! is no communication
                [src[k], dest[k]] = cartcomm.Shift(k, 1)

                # send buffer
                subsizes = np.r_[
                    2, facet_size[:k], overlap_size[k], facet_size[k + 1 :]
                ]
                starts = np.r_[
                    0,
                    np.zeros(k, dtype="i"),
                    sM[k],
                    np.zeros(ndims - k - 1, dtype="i"),
                ]
                sendsubarray.append(
                    MPI.DOUBLE.Create_subarray(
                        sizes, subsizes, starts, order=MPI.ORDER_C
                    )
                )
                resizedsendsubarray.append(
                    sendsubarray[-1].Create_resized(0, overlap_size[k] * itemsize)
                )  # ! see if extent is still fine for more than 2 dimensions
                resizedsendsubarray[-1].Commit()

                # recv buffer
                subsizes = np.r_[
                    2, facet_size[:k], overlap_size[k], facet_size[k + 1 :]
                ]
                starts = np.zeros(ndims + 1, dtype="i")
                recvsubarray.append(
                    MPI.DOUBLE.Create_subarray(
                        sizes, subsizes, starts, order=MPI.ORDER_C
                    )
                )
                resizedrecvsubarray.append(
                    recvsubarray[-1].Create_resized(0, overlap_size[k] * itemsize)
                )
                resizedrecvsubarray[-1].Commit()
            else:
                resizedsendsubarray.append(None)
                resizedrecvsubarray.append(None)
    else:
        for k in range(ndims):
            if overlap_size[k] > 0:
                # ! if there is no overlap along a dimension, make sure there
                # ! is no communication
                [src[k], dest[k]] = cartcomm.Shift(k, -1)

                # recv buffer
                subsizes = np.r_[
                    2, facet_size[:k], overlap_size[k], facet_size[k + 1 :]
                ]
                starts = np.r_[
                    0,
                    np.zeros(k, dtype="i"),
                    sM[k],
                    np.zeros(ndims - k - 1, dtype="i"),
                ]
                recvsubarray.append(
                    MPI.DOUBLE.Create_subarray(
                        sizes, subsizes, starts, order=MPI.ORDER_C
                    )
                )
                resizedrecvsubarray.append(
                    recvsubarray[-1].Create_resized(0, overlap_size[k] * itemsize)
                )  # ! see if extent is still fine for more than 2 dimensions
                resizedrecvsubarray[-1].Commit()

                # send buffer
                subsizes = np.r_[
                    2, facet_size[:k], overlap_size[k], facet_size[k + 1 :]
                ]
                starts = np.zeros(ndims + 1, dtype="i")
                sendsubarray.append(
                    MPI.DOUBLE.Create_subarray(
                        sizes, subsizes, starts, order=MPI.ORDER_C
                    )
                )
                resizedsendsubarray.append(
                    sendsubarray[-1].Create_resized(0, overlap_size[k] * itemsize)
                )
                resizedsendsubarray[-1].Commit()
            else:
                resizedsendsubarray.append(None)
                resizedrecvsubarray.append(None)

    return dest, src, resizedsendsubarray, resizedrecvsubarray


# ! USED IN TESTS ONLY
def mpi_update_borders(
    comm, local_array, dest, src, resizedsendsubarray, resizedrecvsubarray
):
    r"""Update borders of the overlapping facets.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Communicator object.
    local_array : array
        Local buffer, from which data data is sent to (resp. from) workers
        whose rank is given in the list `dest` (resp. `src`).
    dest : list[int]
        List of process ranks to which the current process sends data.
    src : list[int]
        List of process ranks from which the current process receives data.
    resizedsendsubarray : MPI subarray
        Custom MPI subarray type describing the data sent by the current
        process (see `mpi4py.MPI.Datatype.Create_subarray <https://mpi4py.github.io/usrman/reference/mpi4py.MPI.Datatype.html?highlight=create%20subarray#mpi4py.MPI.Datatype.Create_subarray>`_.
    resizedrecvsubarray : MPI subarray
        Custom MPI subarray type describing the data received by the current
        process (see `mpi4py.MPI.Datatype.Create_subarray <https://mpi4py.github.io/usrman/reference/mpi4py.MPI.Datatype.html?highlight=create%20subarray#mpi4py.MPI.Datatype.Create_subarray>`_).

    Note
    ----
    The input array ``local_array`` is updated in-place.
    """

    ndims = len(dest)

    for d in range(ndims):
        comm.Sendrecv(
            [local_array, 1, resizedsendsubarray[d]],
            dest[d],
            recvbuf=[local_array, 1, resizedrecvsubarray[d]],
            source=src[d],
        )

    return


# ! USED IN SANDBOX ONLY
def mpi_circular_update_borders(
    comm,
    local_array,
    dest,
    src,
    resizedsendsubarray,
    resizedrecvsubarray,
    border_resizedsendsubarray,
    border_resizedrecvsubarray,
    ranknd,
    grid_size,
    backward=True,
):
    r"""Update borders of the overlapping facets.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Communicator object.
    local_array : array
        Local buffer, from which data data is sent to (resp. from) workers
        whose rank is given in the list `dest` (resp. `src`).
    dest : list[int]
        List of process ranks to which the current process sends data.
    src : list[int]
        List of process ranks from which the current process receives data.
    resizedsendsubarray : MPI subarray
        Custom MPI subarray type describing the data sent by the current
        process (see `mpi4py.MPI.Datatype.Create_subarray <https://mpi4py.github.io/usrman/reference/mpi4py.MPI.Datatype.html?highlight=create%20subarray#mpi4py.MPI.Datatype.Create_subarray>`_.
    resizedrecvsubarray : MPI subarray
        Custom MPI subarray type describing the data received by the current
        process (see `mpi4py.MPI.Datatype.Create_subarray <https://mpi4py.github.io/usrman/reference/mpi4py.MPI.Datatype.html?highlight=create%20subarray#mpi4py.MPI.Datatype.Create_subarray>`_).
    border_resizedsendsubarray : MPI subarray
        Custom MPI subarray type describing the data received by the first
        worker to the last along each axis (border exchange).
    border_resizedrecvsubarray : MPI subarray
        Custom MPI subarray type describing the data sent by the last worker to
        the first along each axis (border exchange).
    ranknd : numpy.ndarray[int]
        nD rank of the current process in the Cartesian process grid.
    grid_size : numpy.ndarray[int]
        Dimension of the MPI process grid (Cartesian communicator)..

    Note
    ----
    The input array ``local_array`` is updated in-place.
    """

    ndims = len(dest)
    requests = []

    for d in range(ndims):
        comm.Sendrecv(
            [local_array, 1, resizedsendsubarray[d]],
            dest[d],
            recvbuf=[local_array, 1, resizedrecvsubarray[d]],
            source=src[d],
        )

        # ! additional communication (between first and last worker along each
        # ! axis)
        if backward:
            if ranknd[d] == 0:
                requests.append(
                    comm.Isend(
                        [local_array, 1, border_resizedsendsubarray[d]],
                        dest[d],
                    )
                )

            if ranknd[d] == grid_size[d] - 1:
                requests.append(
                    comm.Irecv(
                        [local_array, 1, border_resizedrecvsubarray[d]],
                        src[d],
                    )
                )
        else:
            if ranknd[d] == 0:
                requests.append(
                    comm.Irecv(
                        [local_array, 1, border_resizedrecvsubarray[d]],
                        src[d],
                    )
                )

            if ranknd[d] == grid_size[d] - 1:
                requests.append(
                    comm.Isend(
                        [local_array, 1, border_resizedsendsubarray[d]],
                        dest[d],
                    )
                )

    MPI.Request.Waitall(requests)

    return


# if __name__ == "__main__":
#     # TODO: example to be reactivated / cleansed
#     N = 10
#     nchunks = 3
#     overlap = 1  # number of pixels in the overlap
#     index = 0

#     # no overlap
#     rg0 = split_range(nchunks, N)
#     # check that 2 consecutive entries are distant from 1
#     test1 = np.all(rg0[1:, 0] - rg0[:-1, 1] == 1)
#     # check size of each chunk (uniform in this case)
#     test2 = np.all(np.diff(rg0, n=1, axis=1) + 1 == 3)

#     # # overlap
#     # rg_overlap = split_range(nchunks, N, overlap=overlap)
#     # test1 = np.all(np.abs(rg_overlap[:-1, 1] - rg_overlap[1:, 0] + 1) == overlap)

#     # rgo_r = split_range(nchunks, N, overlap=overlap, backward=False)
#     # test2 = np.all(np.abs(rg_overlap[:-1, 1] - rg_overlap[1:, 0] + 1) == overlap)
#     # rgo_r2 = local_split_range_nd(np.array(2*[nchunks], dtype="i"), np.array(2*[N], dtype="i"), np.array(2*[0], dtype="i"), overlap=np.array(2*[overlap], dtype="i"), backward=False)

#     # # interleaved
#     # rg_interleaved = split_range_interleaved(nchunks, N)
#     # x = np.arange(N)

#     # # start all slices start by k
#     # test_start = np.all(
#     #     [rg_interleaved[k].start == k for k in range(len(rg_interleaved))]
#     # )
#     # test_stop = np.all(
#     #     [rg_interleaved[k].stop == N for k in range(len(rg_interleaved))]
#     # )
#     # test_step = np.all(
#     #     [rg_interleaved[k].step == nchunks for k in range(len(rg_interleaved))]
#     # )

#     # # checking indices for a single process
#     # rg_local = local_split_range(nchunks, N, index, overlap=overlap)
#     # print(rg_local)
#     # rg_local1 = local_split_range_nd(
#     #     np.array([nchunks]),
#     #     np.array([N]),
#     #     np.array([index]),
#     #     overlap=np.array([overlap]),
#     # )
#     # print(np.allclose(rg_local, rg_local1))

#     # index2 = np.array(np.unravel_index(index, 2 * [nchunks]))
#     # rg_local2 = local_split_range_nd(
#     #     np.array(2 * [nchunks]),
#     #     np.array(2 * [N]),
#     #     index2,
#     #     overlap=np.array(2 * [overlap]),
#     # )

#     # rg_local2_sym = local_split_range_symmetric_nd(
#     #     np.array(2 * [nchunks]),
#     #     np.array(2 * [N]),
#     #     index2,
#     #     overlap_pre=np.array(2 * [overlap]),
#     #     overlap_post=np.array(2 * [overlap]),
#     # )

#     # # check consistency between split_range and local version
#     # global_rg = np.concatenate(
#     #     [
#     #         local_split_range(nchunks, N, k, overlap=overlap)[None, :]
#     #         for k in range(nchunks)
#     #     ],
#     #     axis=0,
#     # )
#     # print(global_rg)
#     # err = np.allclose(global_rg, rg_overlap)
#     # print(err)

#     pass
