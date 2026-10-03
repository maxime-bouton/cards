import numpy as np
import pytest
from mpi4py import MPI

import cards.backend as xp
from cards.communicators.mpi_utils import get_ranknd
from cards.operators.distributed_masking import DistributedMasking
from cards.operators.masking import Masking
from cards.slicers.cartesian_comm_slicer import CartesianCommSlicer

# TODO: revise the way rng is handled in MPI test


@pytest.fixture
def mask_loss() -> float:
    return 0.5


@pytest.mark.serial
def test_adjoint(seed, input_shape, mask_loss):
    """Serial test to check the implementation of the adjoint operator is consistent with the direct operator."""

    rng = xp.random.default_rng(seed)
    x = rng.standard_normal(input_shape)
    y = rng.standard_normal(input_shape)
    mask = rng.random(input_shape) < (1 - mask_loss)

    H = Masking(mask)
    Hx = H.forward(x)
    Hadj_y = H.adjoint(y)

    xHadj_y = xp.sum(x * Hadj_y)
    Hxy = xp.sum(Hx * y)

    xp.testing.assert_allclose(Hxy, xHadj_y)


@pytest.mark.mpi
def test_adjoint_mpi(comm, grid_shape, rank, input_shape, seed, mask_loss):
    """Distributed test to check the implementation of the adjoint operator is consistent with the direct operator."""

    grid_size = np.asarray(grid_shape)
    ranknd = get_ranknd(rank, grid_size)
    cartslicer = CartesianCommSlicer(
        ranknd,
        grid_size,
        np.asarray(input_shape),
        np.zeros(len(grid_shape), dtype=int),
        np.zeros(len(grid_shape), dtype=int),
    )
    rng = xp.random.default_rng(seed)
    mask = rng.random(cartslicer.tile_size) < (1 - mask_loss)

    H = DistributedMasking(
        grid_shape,
        input_shape,
        mask,
        cartslicer,
    )

    x = rng.standard_normal(H.cartslicer.tile_size)
    y = rng.standard_normal(H.cartslicer.tile_size)

    Hx = H.forward(x)
    Hadj_y = H.adjoint(y)

    local_Hxy = xp.sum(Hx * y)
    local_xHadj_y = xp.sum(x * Hadj_y)

    Hxy = comm.allreduce(local_Hxy, MPI.SUM)
    xHadj_y = comm.allreduce(local_xHadj_y, MPI.SUM)

    xp.testing.assert_allclose(Hxy, xHadj_y)
