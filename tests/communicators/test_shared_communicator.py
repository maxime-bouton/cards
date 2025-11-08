from mpi4py import MPI

import numpy as np

from mcmc.communicator.shared_communicator import Shared_Communicator
from mcmc.operators.mpi_gradient import MpiGradient2d
from mcmc.operators.mpi_dft_convolution import MpiDftConvolution

import pytest

# TODO np->xp


@pytest.fixture
def seed():
    return 1234


@pytest.fixture
def dims():
    return np.array([6, 124, 75], "i")


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


@pytest.fixture
def kernel_dims():
    return np.array([2, 4, 3], "i")


def test_shared_comm(seed, dims, kernel_dims, comm):
    comm_size = comm.Get_size()
    rank = comm.Get_rank()
    grid_dims = np.asarray(MPI.Compute_dims(comm_size, len(dims)))
    cart_comm = comm.Create_cart(dims=grid_dims)

    rng = np.random.default_rng(seed)

    X = np.zeros(dims)
    kernel = np.zeros(kernel_dims)

    if rank == 0:
        X = rng.standard_normal(dims)
        kernel = rng.standard_normal(kernel_dims)

    cart_comm.Bcast([X, MPI.DOUBLE], root=0)
    cart_comm.Bcast([kernel, MPI.DOUBLE], root=0)

    grad_op = MpiGradient2d(dims, grid_dims, comm)
    convolution_op = MpiDftConvolution(dims, kernel, comm, grid_dims)

    shared_comm = Shared_Communicator(
        comm, grid_dims, dims, {"grad": grad_op, "conv": convolution_op}
    )

    local_X = X[shared_comm.shared_comm.cartslicer._get_slice_global_buffer_to_tile()]

    gradient = grad_op.forward(local_X)
    convolution_product = convolution_op.forward(local_X)

    shared_comm.update_buffer(local_X)

    gradX = shared_comm.apply_operator("grad")
    convX = shared_comm.apply_operator("conv")

    check_grad = np.isclose(gradX, gradient).all
    check_conv = np.isclose(convX, convolution_product).all

    assert check_grad and check_conv
