r"""Testing the MPI-distributed implementation of the 2D divergence operator,
related to the adjoint of the 2D discrete gradient."""

import numpy as np

from mpi4py import MPI

from mcmc.distributed_operators.gradient import distributed_gradient2d
from mcmc.operators.jtv import gradient_2d_adjoint

import pytest

pytestmark = [pytest.mark.mpi, pytest.mark.numpy]


@pytest.fixture
def dims():
    return np.asarray([100, 50], dtype=int)


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


def test_distributed_divergence(comm, dims):
    global_size = dims
    rank = comm.Get_rank()
    grid_dims = np.asarray(MPI.Compute_dims(comm.Get_size(), 2), dtype=int)

    gradient_handler = distributed_gradient2d(global_size, grid_dims)

    m, n = gradient_handler.cart_comm.cartslicer.tile_size

    cart_comm = comm.Create_cart(dims=grid_dims)

    X = np.zeros([2, *dims])

    local_v = np.zeros(gradient_handler.adj_cart_comm_h.cartslicer.tile_size)
    local_h = np.zeros(gradient_handler.adj_cart_comm_v.cartslicer.tile_size)

    if rank == 0:
        X = np.random.rand(2, *dims)

    slice_0 = (
        gradient_handler.adj_cart_comm_h.cartslicer._get_slice_global_buffer_to_tile()
    )
    slice_1 = (
        gradient_handler.adj_cart_comm_v.cartslicer._get_slice_global_buffer_to_tile()
    )

    slices_0 = comm.gather(slice_0, 0)
    slices_1 = comm.gather(slice_1, 0)

    cart_comm.Bcast([X, MPI.DOUBLE], root=0)

    local_h[-m:, -n:] = X[0][slice_0]
    local_v[-m:, -n:] = X[1][slice_1]

    chunk_adj = np.zeros(gradient_handler.adj_cart_comm_h.cartslicer.tile_size)

    gradient_handler.adjoint(chunk_adj, local_h, local_v)

    if rank == 0:
        global_adj = gradient_2d_adjoint(X[0], X[1])

    distributed_adj = np.zeros(dims)

    data = comm.gather(chunk_adj, root=0)

    if rank == 0:
        for i in range(comm.Get_size()):
            distributed_adj[slices_0[i][0], slices_1[i][1]] = data[i]

        assert np.allclose(global_adj, distributed_adj)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    dims = np.asarray([50, 100])

    test_distributed_divergence(comm, dims)
