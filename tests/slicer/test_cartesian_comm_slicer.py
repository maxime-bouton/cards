r"""Test the Cartesian slicer object underlying the communications on a
Cartesian grid of processes. The arrays to be communicated are assumed to be
subarrays of the facet handled by a process.
"""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)

# adpated from: https://gitlab.cristal.univ-lille.fr/pthouven/dsgs

# TODO: add variant for asynchronous communications
# import cards.communicator.async_cartesian_communicator as async_cart_comm
import numpy as np
import pytest
from mpi4py.MPI import PROC_NULL

import cards.communicator.sync_cartesian_communicator as sync_cart_comm
from cards.slicer.cartesian_comm_slicer import (
    CartesianCommSlicer,
    compute_local_buffer_size,
    # create_slice_async_send_recv,
)
from cards.slicer.cartesian_tessellation import local_split_range_nd


@pytest.fixture
def image_size():
    return np.array([20, 20], dtype="i")


@pytest.fixture
def send_size():
    return np.array([2, 3], dtype="i")


@pytest.fixture
def recv_size():
    return np.array([2, 3], dtype="i")


def test_slice_single_worker(image_size, send_size, recv_size):
    r"""Testing slicer on a grid with a single worker (no subarrays sent of received)."""
    grid_size = [1, 1]
    backward = True
    grid_size = np.array(grid_size, dtype="i")
    ndims = image_size.size

    rank = 0
    ranknd = np.array(np.unravel_index(rank, grid_size), dtype="i")

    cart_slicer = CartesianCommSlicer(
        ranknd,
        grid_size,
        image_size,
        send_size,
        recv_size,
        backward=backward,
    )

    facet = np.full(cart_slicer.facet_size, rank, dtype="d")

    assert np.allclose(cart_slicer.facet_size, cart_slicer.tile_size)

    for d in range(ndims):
        assert np.allclose(cart_slicer.recv_size[d], np.zeros((2,), dtype="i"))
        assert np.allclose(cart_slicer.send_size[d], np.zeros((2,), dtype="i"))

        assert np.isclose(cart_slicer.subsizes_recv[d][d], 0)
        assert np.isclose(cart_slicer.subsizes_send[d][d], 0)
        assert np.allclose(facet[cart_slicer.slice_facet_to_tile], facet)


@pytest.mark.parametrize("backward", [True, False])
def test_buffer_size(image_size, recv_size, backward):
    """Tile and facet size should be the same when there is no tessellation along an axis."""
    # ? check issue
    grid_size = np.array([1, 1], dtype="i")
    ranknd = np.zeros((2,), dtype="i")
    tile_range = local_split_range_nd(grid_size, image_size, ranknd)
    tile_size = tile_range[:, 1] - tile_range[:, 0] + 1
    tile_size.astype(int)

    facet_size = compute_local_buffer_size(
        ranknd, grid_size, tile_size, recv_size, backward=backward
    )
    assert np.allclose(facet_size, tile_size)

    # grid_size = np.full((2,), 2, dtype="i")
    # nworkers = np.prod(grid_size)
    # for rank in range(nworkers):
    #     ranknd = np.unravel_index(rank, grid_size)
    #     tile_range = local_split_range_nd(grid_size, image_size, ranknd)
    #     tile_size = tile_range[:, 1] - tile_range[:, 0] + 1
    #     tile_size.astype(int)

    #     facet_size = compute_local_buffer_size(
    #         ranknd, grid_size, tile_size, recv_size, backward=backward
    #     )


def test_fail_negative_send_recv_size(image_size, send_size, recv_size):
    r"""Testing an error is triggered whenever `np.any(recv_size < 0)`
    or `np.any(send_size < 0)` is `True`."""
    ranknd = np.zeros(2, dtype="i")
    grid_size = np.full((2), 2, dtype="i")

    with pytest.raises(ValueError) as excinfo:
        CartesianCommSlicer(
            ranknd,
            grid_size,
            image_size,
            np.array([-1, 2], dtype="i"),
            recv_size,
            backward=True,
        )
    assert "All entries in send_size should be positive." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        CartesianCommSlicer(
            ranknd,
            grid_size,
            image_size,
            send_size,
            np.array([-1, 2], dtype="i"),
            backward=True,
        )
    assert "All entries in recv_size should be positive." in str(excinfo.value)


def test_fail_large_overlap_send(image_size, recv_size):
    r"""Testing an error is triggered whenever `np.any(tile_size > send_size)`
    is `True`."""
    ranknd = np.zeros(2, dtype="i")
    grid_size = np.full((2), 2, dtype="i")

    with pytest.raises(ValueError) as excinfo:
        CartesianCommSlicer(
            ranknd,
            grid_size,
            image_size,
            np.array([15, 11], dtype="i"),
            recv_size,
            backward=True,
        )
    assert "All entries in tile_size should be greater than send_size" in str(
        excinfo.value
    )


def test_1d_backward_recv(image_size, send_size, recv_size):
    r"""Testing slicer on a 3x1 Cartesian grid of workers with backward overlap
    between consecutive facets."""
    nworkers = 3
    grid_size = [3, 1]
    backward = True
    grid_size = np.array(grid_size, dtype="i")

    sum_facets = np.empty(nworkers)
    sum_tiles = np.empty(nworkers)
    sum_recv = np.empty(nworkers)

    expected_sum_facets = np.empty(nworkers)
    expected_sum_tiles = np.empty(nworkers)

    for rank in range(nworkers):
        ranknd = np.array(np.unravel_index(rank, grid_size), dtype="i")

        cart_slicer = CartesianCommSlicer(
            ranknd,
            grid_size,
            image_size,
            send_size,
            recv_size,
            backward=backward,
        )

        facet = np.full(cart_slicer.facet_size, rank + 1, dtype="d")
        # mock communication between workers (backward)
        facet[cart_slicer.slice_recv[0]] = rank

        assert np.allclose(
            np.array(facet[cart_slicer.slice_facet_to_tile].shape, dtype="i"),
            cart_slicer.tile_size,
        )

        sum_facets[rank] = np.sum(facet)
        sum_tiles[rank] = np.sum(facet[cart_slicer.slice_facet_to_tile])
        sum_recv[rank] = np.sum(facet[cart_slicer.slice_recv[0]])

        expected_sum_facets[rank] = (rank + 1) * np.prod(
            cart_slicer.tile_size
        ) + rank * np.size(facet[cart_slicer.slice_recv[0]])

        expected_sum_tiles[rank] = (rank + 1) * np.prod(cart_slicer.tile_size)

    assert np.allclose(sum_facets, expected_sum_facets)
    assert np.allclose(sum_tiles, expected_sum_tiles)
    assert np.allclose(sum_recv, expected_sum_facets - expected_sum_tiles)


def test_1d_forward_recv(image_size, send_size, recv_size):
    r"""Testing slicer on a 3x1 Cartesian grid of workers with forward overlap
    between consecutive facets."""

    nworkers = 3
    grid_size = [3, 1]
    backward = False
    grid_size = np.array(grid_size, dtype="i")

    sum_facets = np.empty(nworkers)
    sum_tiles = np.empty(nworkers)
    sum_recv = np.empty(nworkers)

    expected_sum_facets = np.empty(nworkers)
    expected_sum_tiles = np.empty(nworkers)

    for rank in range(nworkers):
        ranknd = np.array(np.unravel_index(rank, grid_size), dtype="i")

        cart_slicer = CartesianCommSlicer(
            ranknd,
            grid_size,
            image_size,
            send_size,
            recv_size,
            backward=backward,
        )

        facet = np.full(cart_slicer.facet_size, rank + 1, dtype="d")
        # mock communication between workers (forward)
        facet[cart_slicer.slice_recv[0]] = rank + 2

        assert np.allclose(
            np.array(facet[cart_slicer.slice_facet_to_tile].shape, dtype="i"),
            cart_slicer.tile_size,
        )

        sum_facets[rank] = np.sum(facet)
        sum_tiles[rank] = np.sum(facet[cart_slicer.slice_facet_to_tile])
        sum_recv[rank] = np.sum(facet[cart_slicer.slice_recv[0]])

        expected_sum_facets[rank] = (rank + 1) * np.prod(cart_slicer.tile_size) + (
            rank + 2
        ) * np.size(facet[cart_slicer.slice_recv[0]])

        expected_sum_tiles[rank] = (rank + 1) * np.prod(cart_slicer.tile_size)

    assert np.allclose(sum_facets, expected_sum_facets)
    assert np.allclose(sum_tiles, expected_sum_tiles)
    assert np.allclose(sum_recv, expected_sum_facets - expected_sum_tiles)


@pytest.mark.parametrize(
    "backward, nworkers, grid_size, expected_src, expected_dest",
    [
        (
            True,
            4,
            np.array([4], dtype="i"),
            np.array([PROC_NULL, 0, 1, 2], dtype="i"),
            np.array([1, 2, 3, PROC_NULL], dtype="i"),
        ),
        (
            False,
            4,
            np.array([4], dtype="i"),
            np.array([1, 2, 3, PROC_NULL], dtype="i"),
            np.array([PROC_NULL, 0, 1, 2], dtype="i"),
        ),
        (
            True,
            9,
            np.array([3, 3], dtype="i"),
            np.array(
                [
                    [PROC_NULL, PROC_NULL],
                    [PROC_NULL, 0],
                    [PROC_NULL, 1],
                    [0, PROC_NULL],
                    [1, 3],
                    [2, 4],
                    [3, PROC_NULL],
                    [4, 6],
                    [5, 7],
                ],
                dtype="i",
            ),
            np.array(
                [
                    [3, 1],
                    [4, 2],
                    [5, PROC_NULL],
                    [6, 4],
                    [7, 5],
                    [8, PROC_NULL],
                    [PROC_NULL, 7],
                    [PROC_NULL, 8],
                    [PROC_NULL, PROC_NULL],
                ],
                dtype="i",
            ),
        ),
        (
            False,
            9,
            np.array([3, 3], dtype="i"),
            np.array(
                [
                    [3, 1],
                    [4, 2],
                    [5, PROC_NULL],
                    [6, 4],
                    [7, 5],
                    [8, PROC_NULL],
                    [PROC_NULL, 7],
                    [PROC_NULL, 8],
                    [PROC_NULL, PROC_NULL],
                ],
                dtype="i",
            ),
            np.array(
                [
                    [PROC_NULL, PROC_NULL],
                    [PROC_NULL, 0],
                    [PROC_NULL, 1],
                    [0, PROC_NULL],
                    [1, 3],
                    [2, 4],
                    [3, PROC_NULL],
                    [4, 6],
                    [5, 7],
                ],
                dtype="i",
            ),
        ),
    ],
)
def test_sync_send_recv_rank(
    backward, nworkers, grid_size, expected_dest, expected_src
):
    r"""Test index of reception and destination workers for nD synchronous
    communications."""

    ndims = grid_size.size

    src = np.empty((nworkers, ndims), dtype="i")
    dest = np.empty((nworkers, ndims), dtype="i")

    for rank in range(nworkers):
        ranknd = np.array(np.unravel_index(rank, grid_size), dtype="i")
        dest[rank] = sync_cart_comm.send_rank(ranknd, grid_size, backward=backward)
        src[rank] = sync_cart_comm.send_rank(ranknd, grid_size, backward=not backward)

    assert np.allclose(np.squeeze(dest), expected_dest)
    assert np.allclose(np.squeeze(src), expected_src)


# @pytest.mark.parametrize(
#     "backward, nworkers, grid_size, expected_src, expected_dest",
#     [
#         (
#             True,
#             4,
#             np.array([4], dtype="i"),
#             np.array([PROC_NULL, 0, 1, 2], dtype="i"),
#             np.array([1, 2, 3, PROC_NULL], dtype="i"),
#         ),
#         (
#             False,
#             4,
#             np.array([4], dtype="i"),
#             np.array([1, 2, 3, PROC_NULL], dtype="i"),
#             np.array([PROC_NULL, 0, 1, 2], dtype="i"),
#         ),
#         (
#             True,
#             9,
#             np.array([3, 3], dtype="i"),
#             np.array(
#                 [
#                     [PROC_NULL, PROC_NULL, PROC_NULL],
#                     [PROC_NULL, 0, PROC_NULL],
#                     [PROC_NULL, 1, PROC_NULL],
#                     [0, PROC_NULL, PROC_NULL],
#                     [1, 3, 0],
#                     [2, 4, 1],
#                     [3, PROC_NULL, PROC_NULL],
#                     [4, 6, 3],
#                     [5, 7, 4],
#                 ],
#                 dtype="i",
#             ),
#             np.array(
#                 [
#                     [3, 1, 4],
#                     [4, 2, 5],
#                     [5, PROC_NULL, PROC_NULL],
#                     [6, 4, 7],
#                     [7, 5, 8],
#                     [8, PROC_NULL, PROC_NULL],
#                     [PROC_NULL, 7, PROC_NULL],
#                     [PROC_NULL, 8, PROC_NULL],
#                     [PROC_NULL, PROC_NULL, PROC_NULL],
#                 ],
#                 dtype="i",
#             ),
#         ),
#         (
#             False,
#             9,
#             np.array([3, 3], dtype="i"),
#             np.array(
#                 [
#                     [3, 1, 4],
#                     [4, 2, 5],
#                     [5, PROC_NULL, PROC_NULL],
#                     [6, 4, 7],
#                     [7, 5, 8],
#                     [8, PROC_NULL, PROC_NULL],
#                     [PROC_NULL, 7, PROC_NULL],
#                     [PROC_NULL, 8, PROC_NULL],
#                     [PROC_NULL, PROC_NULL, PROC_NULL],
#                 ],
#                 dtype="i",
#             ),
#             np.array(
#                 [
#                     [PROC_NULL, PROC_NULL, PROC_NULL],
#                     [PROC_NULL, 0, PROC_NULL],
#                     [PROC_NULL, 1, PROC_NULL],
#                     [0, PROC_NULL, PROC_NULL],
#                     [1, 3, 0],
#                     [2, 4, 1],
#                     [3, PROC_NULL, PROC_NULL],
#                     [4, 6, 3],
#                     [5, 7, 4],
#                 ],
#                 dtype="i",
#             ),
#         ),
#         # (
#         #     True,
#         #     8,
#         #     np.array([2, 2, 2], dtype="i"),
#         #     np.array(
#         #         [
#         #             [-1, -1, 1, -1, -1, -1, -1],
#         #             [-1, 0, 3, -1, -1, -1, -1],
#         #             [-1, -1, -1, -1, -1, -1, -1],
#         #             [-1, -1, -1, -1, -1, -1, -1],
#         #             [-1, -1, -1, -1, -1, -1, -1],
#         #             [-1, -1, -1, -1, -1, -1, -1],
#         #             [-1, -1, -1, -1, -1, -1, -1],
#         #         ],
#         #         dtype="i",
#         #     ),
#         #     np.array(
#         #         [
#         #             [-1, -1, -1, -1, -1, -1, -1],
#         #             [-1, -1, -1, -1, -1, -1, -1],
#         #             [-1, -1, -1, -1, -1, -1, -1],
#         #             [-1, -1, -1, -1, -1, -1, -1],
#         #             [-1, -1, -1, -1, -1, -1, -1],
#         #             [-1, -1, -1, -1, -1, -1, -1],
#         #             [-1, -1, -1, -1, -1, -1, -1],
#         #         ],
#         #         dtype="i",
#         #     ),
#         # ),
#     ],
# )
# def test_async_send_recv_rank(
#     backward, nworkers, grid_size, expected_dest, expected_src
# ):
#     r"""Test index of reception and destination workers for nD asynchronous
#     communications."""

#     ndims = grid_size.size

#     src = np.squeeze(np.empty((nworkers, ndims * (ndims - 1) + 1), dtype="i"))
#     dest = np.squeeze(np.empty((nworkers, ndims * (ndims - 1) + 1), dtype="i"))

#     for rank in range(nworkers):
#         ranknd = np.array(np.unravel_index(rank, grid_size), dtype="i")
#         dest[rank] = async_cart_comm.send_rank(ranknd, grid_size, backward=backward)
#         src[rank] = async_cart_comm.send_rank(ranknd, grid_size, backward=not backward)

#     assert np.allclose(dest, expected_dest)
#     assert np.allclose(src, expected_src)


# @pytest.mark.parametrize("backward", [True, False])
# @pytest.mark.parametrize(
#     "nworkers, grid_size, imsize, send_size, recv_size",
#     [
#         (
#             4,
#             np.array([4], dtype="i"),
#             np.array([20], dtype="i"),
#             np.array([3], dtype="i"),
#             np.array([2], dtype="i"),
#         ),
#         (
#             9,
#             np.array([3, 3], dtype="i"),
#             np.array([35, 30], dtype="i"),
#             np.array([3, 2], dtype="i"),
#             np.array([2, 1], dtype="i"),
#         ),
#         (
#             8,
#             np.array([2, 2, 2], dtype="i"),
#             np.array([20, 20, 30], dtype="i"),
#             np.array([3, 2, 1], dtype="i"),
#             np.array([2, 2, 2], dtype="i"),
#         ),
#     ],
# )
# def test_slicing_async(
#     backward, nworkers, grid_size, imsize, send_size, recv_size
# ):
#     r"""Test slicing to extract sent and received entries for asynchronous
#     communication phases."""
#     ndims = grid_size.size

#     cart_slicer = nworkers * [None]
#     facet = nworkers * [None]
#     s_rank = nworkers * [None]
#     r_rank = nworkers * [None]
#     async_slice_recv = nworkers * [None]
#     async_slice_send = nworkers * [None]
#     async_starts_send = nworkers * [None]
#     async_subsizes_send = nworkers * [None]
#     async_starts_recv = nworkers * [None]
#     async_subsizes_recv = nworkers * [None]

#     for rank in range(nworkers):
#         ranknd = np.array([rank], dtype="i")
#         ranknd = np.array(np.unravel_index(rank, grid_size), dtype="i")

#         cart_slicer[rank] = CartesianCommSlicer(
#             ranknd,
#             grid_size,
#             imsize,
#             send_size,
#             recv_size,
#             backward=backward,
#         )

#         (
#             async_slice_send[rank],
#             async_slice_recv[rank],
#             async_starts_send[rank],
#             async_subsizes_send[rank],
#             async_starts_recv[rank],
#             async_subsizes_recv[rank],
#         ) = create_slice_async_send_recv(
#             ranknd,
#             grid_size,
#             cart_slicer[rank].facet_size,
#             cart_slicer[rank].send_size,
#             cart_slicer[rank].recv_size,
#             backward=backward,
#         )

#         # rank of the workers to which the current process should send info
#         s_rank[rank] = async_cart_comm.send_rank(ranknd, grid_size, backward=backward)

#         # rank of the workers from which the current process should receive info
#         r_rank[rank] = async_cart_comm.send_rank(
#             ranknd, grid_size, backward=not backward
#         )

#         # testing shape + mock synchronous communication between workers
#         facet[rank] = np.empty(cart_slicer[rank].facet_size)
#         mask = np.full(ndims, False)
#         expected_shape_recv = cart_slicer[rank].tile_size.copy()
#         expected_shape_send = cart_slicer[rank].tile_size.copy()

#         for k in range(ndims - 1):
#             mask[k] = True
#             for d in range(ndims):
#                 expected_shape_recv[mask] = cart_slicer[rank].recv_size[mask]
#                 expected_shape_send[mask] = cart_slicer[rank].send_size[mask]

#                 assert np.allclose(
#                     np.array(
#                         facet[rank][async_slice_recv[rank][k * ndims + d]].shape,
#                         dtype="i",
#                     ),
#                     expected_shape_recv,
#                 )
#                 assert np.allclose(
#                     async_subsizes_recv[rank][k * ndims + d],
#                     expected_shape_recv,
#                 )

#                 assert np.allclose(
#                     np.array(
#                         facet[rank][async_slice_send[rank][k * ndims + d]].shape,
#                         dtype="i",
#                     ),
#                     expected_shape_send,
#                 )
#                 assert np.allclose(
#                     async_subsizes_send[rank][k * ndims + d],
#                     expected_shape_send,
#                 )

#                 expected_shape_recv[mask] = cart_slicer[rank].tile_size[mask]
#                 expected_shape_send[mask] = cart_slicer[rank].tile_size[mask]
#                 mask = np.roll(mask, 1)

#                 # mock communication (copying rank value from nearby process)
#                 if r_rank[rank][k * ndims + d] > PROC_NULL:
#                     facet[rank][async_slice_recv[rank][k * ndims + d]] = r_rank[rank][
#                         k * ndims + d
#                     ]

#         expected_shape_recv = cart_slicer[rank].recv_size
#         assert np.allclose(
#             np.array(facet[rank][async_slice_recv[rank][-1]].shape, dtype="i"),
#             expected_shape_recv,
#         )
#         assert np.allclose(
#             async_subsizes_recv[rank][-1],
#             expected_shape_recv,
#         )

#         expected_shape_send = cart_slicer[rank].send_size
#         assert np.allclose(
#             np.array(facet[rank][async_slice_send[rank][-1]].shape, dtype="i"),
#             expected_shape_send,
#         )
#         assert np.allclose(
#             async_subsizes_send[rank][-1],
#             expected_shape_send,
#         )

#         # mock communication (copying rank value from nearby process)
#         if r_rank[rank][-1] > PROC_NULL:
#             facet[rank][async_slice_recv[rank][-1]] = r_rank[rank][-1]

#         # check values in facets and ranks of workers
#         for q in range(ndims * (ndims - 1) + 1):
#             if r_rank[rank][q] > PROC_NULL:
#                 assert np.allclose(
#                     facet[rank][async_slice_recv[rank][q]], r_rank[rank][q]
#                 )


@pytest.mark.parametrize("backward", [True, False])
@pytest.mark.parametrize(
    "nworkers, grid_size, imsize, send_size, recv_size",
    [
        (
            4,
            np.array([4], dtype="i"),
            np.array([20], dtype="i"),
            np.array([3], dtype="i"),
            np.array([2], dtype="i"),
        ),
        (
            9,
            np.array([3, 3], dtype="i"),
            np.array([35, 30], dtype="i"),
            np.array([3, 2], dtype="i"),
            np.array([2, 1], dtype="i"),
        ),
        (
            8,
            np.array([2, 2, 2], dtype="i"),
            np.array([20, 20, 30], dtype="i"),
            np.array([3, 2, 1], dtype="i"),
            np.array([2, 2, 2], dtype="i"),
        ),
    ],
)
def test_slicing_sync(backward, nworkers, grid_size, imsize, send_size, recv_size):
    r"""Test slicing to extract sent and received entries for synchronous
    communication phases (1 along each axis of the process grid)."""
    ndims = grid_size.size

    for rank in range(nworkers):
        ranknd = np.array([rank], dtype="i")
        ranknd = np.array(np.unravel_index(rank, grid_size), dtype="i")

        cart_slicer = CartesianCommSlicer(
            ranknd,
            grid_size,
            imsize,
            send_size,
            recv_size,
            backward=backward,
        )

        # testing shape + mock synchronous communication between workers
        facet = np.empty(cart_slicer.facet_size)

        r_size = cart_slicer.facet_size.copy()
        s_size = cart_slicer.facet_size.copy()

        for d in range(ndims):
            r_size[d] = cart_slicer.recv_size[d]
            assert np.allclose(
                np.array(
                    facet[cart_slicer.slice_recv[d]].shape,
                    dtype="i",
                ),
                r_size,
            )
            assert np.allclose(cart_slicer.subsizes_recv[d], r_size)
            r_size[d] = cart_slicer.facet_size[d]

            s_size[d] = cart_slicer.send_size[d]
            assert np.allclose(
                np.array(
                    facet[cart_slicer.slice_send[d]].shape,
                    dtype="i",
                ),
                s_size,
            )
            assert np.allclose(cart_slicer.subsizes_send[d], s_size)
            s_size[d] = cart_slicer.facet_size[d]
