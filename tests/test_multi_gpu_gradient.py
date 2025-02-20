"""Check if the computations done with the distributed implementation of the gradient operator correspond to the serial one."""

import cupy as cp
from mpi4py import MPI
import numpy as np

from mcmc.distributed_operators.multi_gpu.gradient import (
    distributed_gradient2d as gpu_grad,
)
from mcmc.distributed_operators.gradient import distributed_gradient2d as mpi_grad

import pytest


@pytest.fixture
def seed():
    return 1234


@pytest.fixture
def dims():
    return np.asarray([100, 50])


def test_gradient_multi_gpu(seed, dims):
    X = cp.zeros(dims)

    rng = np.random.default_rng(seed)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    grid_dims = np.asarray(MPI.Compute_dims(comm_size, 2))

    with cp.cuda.Device(rank):
        X = cp.zeros(dims)
    if rank == 0:
        X = cp.asarray(rng.standard_normal(dims))

    with cp.cuda.Device(rank):
        X = comm.bcast(X, root=0)

    gpu_gradient_handler = gpu_grad(dims, grid_dims, comm)
    cpu_gradient_handler = mpi_grad(dims, grid_dims, comm)

    slice = gpu_gradient_handler.cart_comm.cartslicer._get_slice_global_buffer_to_tile()

    local_X = X[slice]

    local_gpu_grad = gpu_gradient_handler.forward(local_X)
    local_cpu_grad = cpu_gradient_handler.forward(cp.asnumpy(local_X))

    local_check = cp.allclose(local_gpu_grad, cp.asarray(local_cpu_grad))

    global_check = comm.reduce(local_check, MPI.PROD, root=0)

    if rank == 0:
        assert global_check


def test_gradient_adjoint_multi_gpu(seed, dims):
    X = cp.zeros(dims)

    rng = np.random.default_rng(seed)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    grid_dims = np.asarray(MPI.Compute_dims(comm_size, 2))
    rank = comm.Get_rank()

    with cp.cuda.Device(rank):
        X = cp.zeros((*dims, 2))
    if rank == 0:
        X = cp.asarray(rng.standard_normal((*dims, 2)))

    with cp.cuda.Device(rank):
        X = comm.bcast(X, root=0)

    gpu_gradient_handler = gpu_grad(dims, grid_dims, comm)
    cpu_gradient_handler = mpi_grad(dims, grid_dims, comm)

    slice = gpu_gradient_handler.cart_comm.cartslicer._get_slice_global_buffer_to_tile()

    local_X = X[slice]

    local_cpu_adj = np.zeros(cpu_gradient_handler.cart_comm.cartslicer.tile_size)
    local_gpu_adj = cp.zeros(gpu_gradient_handler.cart_comm.cartslicer.tile_size)
    local_gpu_adj = gpu_gradient_handler.adjoint(local_X[:, :, 0], local_X[:, :, 1])
    cpu_gradient_handler.adjoint(
        local_cpu_adj, cp.asnumpy(local_X)[:, :, 0], cp.asnumpy(local_X)[:, :, 1]
    )

    local_check = cp.allclose(local_gpu_adj, cp.asarray(local_cpu_adj))

    global_check = comm.reduce(local_check, MPI.PROD, root=0)

    if rank == 0:
        assert global_check


if __name__ == "__main__":
    dims = np.asarray([100, 50])
    seed = 1234

    test_gradient_multi_gpu(seed, dims)
    test_gradient_adjoint_multi_gpu(seed, dims)
