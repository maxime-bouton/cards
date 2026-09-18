import numpy as np
import pytest
from mpi4py import MPI

import cards.backend as xp
from cards.operators.distributed_gradient import DistributedGradient2d
from cards.operators.gradient import Gradient2d

# FIxME: cleanse distributed test, avoid copying full array on all workers


@pytest.mark.serial
def test_basic_check(input_shape):
    """
    Test that the gradient of a constant array is zero.
    """
    x = xp.ones(input_shape)
    G = Gradient2d(input_shape)
    Gx = G.forward(x)

    assert xp.allclose(Gx, 0)


@pytest.mark.serial
def test_adjoint(seed, input_shape):
    """
    Test the adjoint property of the gradient operator in serial setting.
    """
    rng = xp.random.default_rng(seed)
    x = rng.standard_normal(input_shape)
    y = rng.standard_normal((2, *input_shape))

    G = Gradient2d(input_shape)
    Hx = G.forward(x)
    Hy = G.adjoint(y)

    xHy = xp.sum(x * Hy)
    Hxy = xp.sum(Hx * y)

    xp.testing.assert_allclose(Hxy, xHy)


@pytest.mark.mpi
def test_adjoint_mpi(comm, input_shape, seed):
    """
    Test the adjoint property of the gradient operator in MPI setting.
    """
    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    grid_dims = (1, *MPI.Compute_dims(comm_size, 2))
    cart_comm = comm.Create_cart(dims=grid_dims)

    G = DistributedGradient2d(input_shape, grid_dims, comm)

    rng = xp.random.default_rng(seed)

    x = xp.zeros(input_shape)
    y = xp.zeros((2, *input_shape))

    if rank == 0:
        x = rng.standard_normal(input_shape)
        y = rng.standard_normal((2, *input_shape))
    cart_comm.Bcast([x, MPI.DOUBLE], root=0)
    cart_comm.Bcast([y, MPI.DOUBLE], root=0)

    local_slice = G.direct_communicator.cartslicer.slice_global_buffer_to_tile

    local_x = x[local_slice]
    local_y = xp.zeros((2, *G.adjoint_communicator_h.cartslicer.tile_size))
    local_slice_h = G.adjoint_communicator_h.cartslicer.slice_global_buffer_to_tile
    local_slice_v = G.adjoint_communicator_v.cartslicer.slice_global_buffer_to_tile

    slice_h = np.s_[0, *local_slice_h]
    slice_v = np.s_[1, *local_slice_v]
    local_y[0] = y[slice_h]
    local_y[1] = y[slice_v]

    local_grad = G.forward(local_x)
    local_adj = G.adjoint(local_y)

    local_Hxy = xp.sum(local_grad[0] * local_y[0] + local_grad[1] * local_y[1])
    local_xHy = xp.sum(x[local_slice] * local_adj)

    Hxy = comm.allreduce(local_Hxy, MPI.SUM)
    xHy = comm.allreduce(local_xHy, MPI.SUM)

    xp.testing.assert_allclose(Hxy, xHy)
