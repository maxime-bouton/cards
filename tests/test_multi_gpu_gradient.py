import cupy as cp
from mpi4py import MPI
import numpy as np

from mcmc.distributed_operators.multi_gpu.gradient import distributed_gradient2d
from mcmc.operators.gradient import Gradient2d

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

    if rank == 0:
        X = cp.asarray(rng.standard_normal(dims))

    comm.bcast(X, root=0)

    gradient_handler = distributed_gradient2d(dims, grid_dims)

    slice = gradient_handler.cart_comm.cartslicer._get_slice_global_buffer_to_tile()

    local_X = X[slice]

    local_grad = gradient_handler.forward(local_X)

    full_grad_handler = Gradient2d(dims)
    grad = full_grad_handler.forward(cp.asnumpy(X))

    local_check = cp.allclose(local_grad, cp.asarray(grad[slice, slice]))

    global_check = comm.reduce(local_check, MPI.PROD, root=0)

    if rank == 0:
        assert global_check
