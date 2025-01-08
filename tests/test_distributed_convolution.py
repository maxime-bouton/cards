"""Check if the computations done with the distributed implementation of the convolution operator correspond to the serial one."""

import numpy as np
from mpi4py import MPI

import pytest

from mcmc.distributed_operators.sync_linear_convolution import SyncLinearConvolution
from mcmc.operators.serial_convolution import fft_conv
from mcmc.slicer.cartesian_comm_slicer import CartesianCommSlicer


@pytest.fixture
def seed():
    return 1234


@pytest.fixture
def dims():
    return np.asarray([100, 50])


@pytest.fixture
def kernel_dims():
    return np.asarray([5, 5])


def test_distributed_convolution_forward(seed, dims, kernel_dims):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()

    grid_dims = np.asarray(MPI.Compute_dims(comm_size, 2))
    cart_comm = comm.Create_cart(grid_dims)
    ranknd = np.asarray(cart_comm.Get_coords(rank))
    slicer = CartesianCommSlicer(
        ranknd, grid_dims, dims, np.asarray([0, 0]), np.asarray([0, 0])
    )

    X = np.zeros(dims)
    kernel = np.zeros(kernel_dims)

    if rank == 0:
        rng = np.random.default_rng(seed)
        X = rng.standard_normal(dims)
        kernel = rng.standard_normal(kernel_dims)

    comm.Bcast([X, MPI.DOUBLE], root=0)
    comm.Bcast([kernel, MPI.DOUBLE], root=0)

    convo_dims = dims + kernel_dims - np.ones_like(dims)

    Y = fft_conv(
        X, np.fft.rfftn(kernel, convo_dims, axes=range(len(convo_dims))), convo_dims
    )

    local_X = X[slicer.slice_global_buffer_to_tile]

    convolution_handler = SyncLinearConvolution(dims, kernel, comm, grid_dims)
    local_Y = convolution_handler.forward(local_X)

    slice = convolution_handler.adjoint_communicator.cartslicer._get_slice_global_buffer_to_tile()

    local_check = np.allclose(Y[slice], local_Y, atol=1e-10)

    global_check = comm.reduce(local_check, MPI.PROD, root=0)

    if rank == 0:
        assert global_check


def test_distributed_convolution_adjoint(seed, dims, kernel_dims):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()

    grid_dims = np.asarray(MPI.Compute_dims(comm_size, 2))
    cart_comm = comm.Create_cart(grid_dims)
    ranknd = np.asarray(cart_comm.Get_coords(rank))

    X = np.zeros(dims + kernel_dims - np.ones_like(dims))
    kernel = np.zeros(kernel_dims)

    if rank == 0:
        rng = np.random.default_rng(seed)
        X = rng.standard_normal(dims + kernel_dims - np.ones_like(dims))
        kernel = rng.standard_normal(kernel_dims)

    comm.Bcast([X, MPI.DOUBLE], root=0)
    comm.Bcast([kernel, MPI.DOUBLE], root=0)

    adj_dims = dims + kernel_dims - np.ones_like(dims)

    Y = fft_conv(
        X, np.conj(np.fft.rfftn(kernel, adj_dims, axes=range(len(adj_dims)))), adj_dims
    )

    convolution_handler = SyncLinearConvolution(dims, kernel, comm, grid_dims)
    local_X = X[
        convolution_handler.adjoint_communicator.cartslicer.slice_global_buffer_to_tile
    ]
    local_Y = convolution_handler.adjoint(local_X)

    slicer = CartesianCommSlicer(
        ranknd, grid_dims, dims, np.asarray([0, 0]), np.asarray([0, 0])
    )

    slice = slicer.slice_global_buffer_to_tile
    local_check = np.allclose(Y[slice], local_Y)

    global_check = comm.reduce(local_check, MPI.PROD, root=0)

    assert local_check
    if rank == 0:
        assert global_check


if __name__ == "__main__":
    seed = 1234
    dims = np.asarray([100, 50])
    kernel_dims = np.asarray([5, 5])

    test_distributed_convolution_forward(seed, dims, kernel_dims)
    test_distributed_convolution_adjoint(seed, dims, kernel_dims)
