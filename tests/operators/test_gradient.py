r"""Testing implementation consistency and error messages for discrete gradient operators."""

import pytest
from mpi4py import MPI

import cards.backend as xp
from cards.operators.distributed_gradient import DistributedGradient2d
from cards.operators.gradient import Gradient2d

# TODO: revise the way rng is handled in MPI test


@pytest.mark.serial
def test_basic_check(input_shape):
    """Test that the gradient of a constant array is zero."""
    x = xp.ones(input_shape)
    H = Gradient2d(input_shape)
    Hx = H.forward(x)

    assert xp.allclose(Hx, 0)


@pytest.mark.serial
def test_adjoint(seed, input_shape):
    """Serial test to check the implementation of the adjoint operator is consistent with the direct operator."""
    rng = xp.random.default_rng(seed)
    x = rng.standard_normal(input_shape)
    y = rng.standard_normal((2, *input_shape))

    H = Gradient2d(input_shape)
    Hx = H.forward(x)
    Hadj_y = H.adjoint(y)

    xHadj_y = xp.sum(x * Hadj_y)
    Hxy = xp.sum(Hx * y)

    xp.testing.assert_allclose(Hxy, xHadj_y)


@pytest.mark.mpi
def test_adjoint_mpi(comm, input_shape, grid_shape, seed):
    """Distributed test to check the implementation of the adjoint operator is consistent with the direct operator."""
    H = DistributedGradient2d(input_shape, grid_shape, comm)

    rng = xp.random.default_rng(seed)
    x = rng.standard_normal(H.direct_communicator.cartslicer.tile_size)
    y = rng.standard_normal(H.adjoint_tile_size)

    Hx = H.forward(x)
    Hadj_y = H.adjoint(y)

    local_Hxy = xp.sum(Hx * y)
    local_xHadj_y = xp.sum(x * Hadj_y)

    Hxy = comm.allreduce(local_Hxy, MPI.SUM)
    xHadj_y = comm.allreduce(local_xHadj_y, MPI.SUM)

    xp.testing.assert_allclose(Hxy, xHadj_y)
