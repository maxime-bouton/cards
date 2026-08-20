import numpy as np
import pytest
from mpi4py import MPI

import cards.backend as xp
from cards.operators.gradient import Gradient2d
from cards.operators.mpi_gradient import MpiGradient2d


@pytest.mark.serial
def test_basic_check(input_shape):
    """
    Test that the gradient of a constant array is zero.
    """
    X = xp.ones(input_shape)
    gradient_operator = Gradient2d(input_shape)

    grad = gradient_operator.forward(X)

    assert xp.allclose(grad, 0)


@pytest.mark.serial
def test_adjoint(seed, input_shape):
    """
    Test the adjoint property of the gradient operator in serial setting.
    """
    rng = xp.random.default_rng(seed)
    X = rng.standard_normal(input_shape)
    Y = rng.standard_normal((2, *input_shape))

    grad_op = Gradient2d(input_shape)
    Hx = grad_op.forward(X)
    Hy = grad_op.adjoint(Y)

    xHy = xp.sum(X * Hy)
    Hxy = xp.sum(Hx * Y)

    xp.testing.assert_allclose(Hxy, xHy)


# FIXME: cleanse test, avoid copying full array on all workers
@pytest.mark.mpi
def test_adjoint_mpi(comm, input_shape, seed):
    """
    Test the adjoint property of the gradient operator in MPI setting.
    """
    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    grid_dims = (1, *MPI.Compute_dims(comm_size, 2))
    cart_comm = comm.Create_cart(dims=grid_dims)

    grad_op = MpiGradient2d(input_shape, grid_dims, comm)

    rng = xp.random.default_rng(seed)

    X = xp.zeros(input_shape)
    Y = xp.zeros((2, *input_shape))

    if rank == 0:
        X = rng.standard_normal(input_shape)
        Y = rng.standard_normal((2, *input_shape))
    cart_comm.Bcast([X, MPI.DOUBLE], root=0)
    cart_comm.Bcast([Y, MPI.DOUBLE], root=0)

    local_slice = grad_op.cart_comm.cartslicer.slice_global_buffer_to_tile

    local_X = X[local_slice]
    local_Y = xp.zeros((2, *grad_op.adj_cart_comm_h.cartslicer.tile_size))
    local_adj = xp.zeros(grad_op.adj_cart_comm_h.cartslicer.tile_size)
    local_slice_h = grad_op.adj_cart_comm_h.cartslicer.slice_global_buffer_to_tile
    local_slice_v = grad_op.adj_cart_comm_v.cartslicer.slice_global_buffer_to_tile

    slice_h = np.s_[0, *local_slice_h]
    slice_v = np.s_[1, *local_slice_v]
    local_Y[0] = Y[slice_h]
    local_Y[1] = Y[slice_v]

    local_grad = grad_op.forward(local_X)
    local_adj = grad_op.adjoint(local_Y)

    local_Hxy = xp.sum(local_grad[0] * local_Y[0] + local_grad[1] * local_Y[1])
    local_xHy = xp.sum(X[local_slice] * local_adj)

    Hxy = 0
    xHy = 0
    Hxy = comm.allreduce(local_Hxy, MPI.SUM)
    xHy = comm.allreduce(local_xHy, MPI.SUM)

    xp.testing.assert_allclose(Hxy, xHy)
