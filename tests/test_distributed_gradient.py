r"""Testing the MPI-distributed implementation of the 2D discrete gradient
operator."""

import numpy as np
import pytest
from mpi4py import MPI

from mcmc.distributed_operators.gradient import distributed_gradient2d
from mcmc.operators.jtv import gradient_2d

pytestmark = [pytest.mark.mpi, pytest.mark.numpy]


@pytest.fixture
def dims():
    return np.asarray([100, 50], dtype=int)


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


@pytest.mark.numpy
def test_distributed_gradient(comm, dims):
    global_size = dims
    rank = comm.Get_rank()
    grid_dims = np.asarray(MPI.Compute_dims(comm.Get_size(), 2), dtype=int)

    gradient_handler = distributed_gradient2d(global_size, grid_dims)

    cart_comm = comm.Create_cart(dims=grid_dims)

    X = np.zeros(dims)

    if rank == 0:
        X = np.random.rand(*dims)

    cart_comm.Bcast([X, MPI.DOUBLE], root=0)

    slice_0 = gradient_handler.cart_comm.cartslicer._get_slice_global_buffer_to_tile()[
        0
    ]
    slice_1 = gradient_handler.cart_comm.cartslicer._get_slice_global_buffer_to_tile()[
        1
    ]

    slices_0_start = comm.gather(slice_0.start, 0)
    slices_0_end = comm.gather(slice_0.stop, 0)

    slices_1_start = comm.gather(slice_1.start, 0)
    slices_1_end = comm.gather(slice_1.stop, 0)

    local = np.zeros(gradient_handler.cart_comm.cartslicer.tile_size)
    local = X[slice_0, slice_1]

    chunk_grad = gradient_handler.forward(local)

    global_grad = np.zeros(shape=[2, *dims])
    distributed_grad = np.zeros(shape=[2, *dims])

    if rank == 0:
        global_grad = gradient_2d(X)

    data = comm.gather(chunk_grad, root=0)

    if rank == 0:
        for i in range(comm.Get_size()):
            distributed_grad[
                :,
                slices_0_start[i] : slices_0_end[i],
                slices_1_start[i] : slices_1_end[i],
            ] = data[i]

        assert np.allclose(global_grad, distributed_grad)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    dims = np.asarray([50, 100])

    test_distributed_gradient(comm, dims)
