""" Set of helper functions to implement a MPI-distributed convolution operator
and its adjoint.
"""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# reference: P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.

import numpy as np


def calculate_local_data_size(
    tile_size, ranknd, overlap_size, grid_size, backward=True
):
    r"""Compute the size of the chunk of convolution hold by the current
    worker.

    Parameters
    ----------
    tile_size : numpy.ndarray[int]
        Size of the non-overlapping image tile underlying the overlapping
        facets.
    ranknd : numpy.ndarray[int]
        Rank of the current process in the nD grid of MPI processes.
    overlap_size : numpy.ndarray[int]
        Size of the overlap along each dimension.
    grid_size : numpy.ndarray[int]
        Number of processes along each dimension of the nD MPI process grid.
    backward : bool, optional
        Direction of the overlap in the cartesian grid along all the
        dimensions (forward or backward overlap), by default True.

    Returns
    -------
    local_data_size : numpy.ndarray[int]
        Size of the local chunk of the data owned by the current process.
    facet_size : numpy.ndarray[int]
        Size of the overlapping facet handled by the current process (direct
        operator).
    facet_size_adj : numpy.ndarray[int]
        Size of the overlapping facet handled by the current process (adjoint
        operator).
    """
    # TODO: allow overlap only if a dimension is split into several chunks
    if backward:
        local_data_size = tile_size + (ranknd == grid_size - 1) * overlap_size
        facet_size = tile_size + (ranknd > 0) * overlap_size
        facet_size_adj = local_data_size + (ranknd < grid_size - 1) * overlap_size
    else:
        local_data_size = tile_size + (ranknd == 0) * overlap_size
        facet_size = tile_size + (ranknd < grid_size - 1) * overlap_size
        facet_size_adj = local_data_size + (ranknd > 0) * overlap_size

    return local_data_size, facet_size, facet_size_adj


def create_local_to_global_slice(
    tile_pixels, ranknd, overlap_size, local_data_size, backward=True
):
    r"""Create a slice object to place a local chunk of the convolution into
    the full array structure.

    Parameters
    ----------
    tile_pixels : numpy.ndarray[int]
        Index of the first pixel into non-overlapping pixel tile.
    ranknd : numpy.ndarray[int]
        Rank of the current process in the nD grid of MPI processes.
    overlap_size : numpy.ndarray[int]
        Size of the overlap along each dimension.
    local_data_size : numpy.ndarray[int]
        Size of the local chunk of the data owned by the current process.
    backward : bool, optional
        Direction of the overlap in the cartesian grid along all the
        dimensions (forward or backward overlap), by default True.

    Returns
    -------
    global_slice : tuple[slice]
        Tuple slice to place the local chunk of convolution data into an array
        representing the full convolution.
    """

    ndims = np.size(ranknd)

    # ! offset required only when using forward overlap
    offset = np.logical_and(not backward, ranknd > 0) * overlap_size

    global_slice = tuple(
        [
            np.s_[
                tile_pixels[d, 0]
                + offset[d] : tile_pixels[d, 0]
                + offset[d]
                + local_data_size[d]
            ]
            for d in range(ndims)
        ]
    )

    return global_slice
