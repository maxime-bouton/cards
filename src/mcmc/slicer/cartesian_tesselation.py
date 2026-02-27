"""Utility functions to compute the indices of each subarray involved in a
Cartesian domain decomposition."""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)

import numpy as np


def get_neighbour(ranknd, grid_size, disp):
    """Linear rank of a neighbour of the current MPI process.

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
        Linear rank of the neighbour process
    """

    return np.ravel_multi_index(ranknd + disp, grid_size)


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

    Return the portion of :math:`\{ 0, \dotsc , N-1 \}`, tesselated into
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

    if nchunks <= 1:
        start = -1
    else:
        start = -1 + index * step
    stop = np.rint(start + step)
    start = np.rint(start)
    rg = np.array([start + 1, stop], dtype="i")
    # if the facet overlaps with a neighbour
    if overlap > 0 and nchunks > 1:
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

    Return the portion of :math:`\{ 0, \dotsc , N-1 \}`, tesselated
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

    # making function compatible with a 1D implementation:
    # avoid error for 1D communications with 2D arrays, requiring
    # change in range generation if no splitting along some axes
    id_err_index = nchunks > 1

    if np.any(id_err_index):
        if np.any(nchunks[id_err_index] <= index):
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
    start[nchunks <= 1] = -1
    stop = (start + step).astype(np.int64)
    start = start.astype(np.int64)
    rg = np.concatenate(((start + 1)[:, None], stop[:, None]), axis=1)

    if overlap is not None:
        if backward:
            sel = np.logical_and(index > 0, overlap > 0)
            rg[sel, 0] = rg[sel, 0] - overlap[sel]
        else:
            sel = np.logical_and(index < nchunks - 1, overlap > 0)
            rg[sel, 1] = rg[sel, 1] + overlap[sel]
    return rg


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
