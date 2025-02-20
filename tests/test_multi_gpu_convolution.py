"""Check if the computations done with the distributed implementation of the convolution operator correspond to the serial one."""

import numpy as np
import cupy as cp
from mpi4py import MPI

import pytest

from mcmc.distributed_operators.multi_gpu.dft_convolution import MultiGPU_DFTConvolution
from mcmc.operators.gpu.convolution import fft_conv
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


def test_multi_gpu_convolution_forward(seed, dims, kernel_dims):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()

    grid_dims = np.asarray(MPI.Compute_dims(comm_size, 2))
    cart_comm = comm.Create_cart(grid_dims)
    ranknd = np.asarray(cart_comm.Get_coords(rank))
    slicer = CartesianCommSlicer(
        ranknd, grid_dims, dims, np.asarray([0, 0]), np.asarray([0, 0])
    )

    X = cp.zeros(dims)
    kernel = cp.zeros(kernel_dims)

    if rank == 0:
        rng = cp.random.default_rng(seed)
        X = rng.standard_normal(dims)
        kernel = rng.standard_normal(kernel_dims)

    comm.Bcast([X, MPI.DOUBLE], root=0)
    comm.Bcast([kernel, MPI.DOUBLE], root=0)

    convo_dims = dims + kernel_dims - np.ones_like(dims)

    Y = fft_conv(
        X,
        np.fft.rfftn(kernel, tuple(convo_dims), axes=range(len(convo_dims))),
        tuple(convo_dims),
    )

    local_X = X[slicer.slice_global_buffer_to_tile]

    convolution_handler = MultiGPU_DFTConvolution(dims, kernel, comm, grid_dims)
    local_Y = convolution_handler.forward(local_X)

    slice = convolution_handler.adjoint_communicator.cartslicer._get_slice_global_buffer_to_tile()

    local_check = np.allclose(Y[slice], local_Y, atol=1e-10)

    global_check = comm.reduce(local_check, MPI.PROD, root=0)

    if rank == 0:
        assert global_check


if __name__ == "__main__":
    seed = 1234
    dims = np.asarray([100, 50])
    kernel_dims = np.asarray([5, 5])

    test_multi_gpu_convolution_forward(seed, dims, kernel_dims)

# mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m mpi4py -m pytest test_multi_gpu_convolution
