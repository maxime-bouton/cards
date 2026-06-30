"""Tests for `SharedCommunicator` class."""

import numpy as np
import pytest
from mpi4py import MPI

import cards.backend as xp
from cards.communicators.shared_communicator import SharedCommunicator
from cards.operators.mpi_dft_convolution import MpiDftConvolution
from cards.operators.mpi_gradient import MpiGradient2d
from cards.utils.utils import expand_shape_left


@pytest.fixture
def kernel_shape() -> tuple[int, ...]:
    return (5, 3)


@pytest.fixture(params=[1, 2])
def grid_ndim(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def grid_shape(
    comm: MPI.Comm,
    grid_ndim: int,
    input_shape: tuple[int, ...],
) -> tuple[int, ...]:
    return expand_shape_left(
        MPI.Compute_dims(comm.Get_size(), grid_ndim),
        ndim=len(input_shape),
    )


@pytest.mark.mpi
def test_shared_comm(
    comm: MPI.Comm,
    grid_shape: tuple[int, ...],
    input_shape: tuple[int, ...],
    kernel_shape: tuple[int, ...],
    seed: int,
) -> None:
    """
    Verify `SharedCommunicator` yields results identical to individual communicators.

    Check consistency across discrete Total Variation (TV) and
    DFT convolution operators.
    """
    input_size = np.asarray(input_shape)
    grid_size = np.asarray(grid_shape)

    # ensures kernel and input have same number of dimensions
    kernel_size = np.asarray(expand_shape_left(kernel_shape, ndim=len(input_shape)))

    rng = xp.random.default_rng(seed)
    full_x = rng.random(input_shape, dtype=xp.float32)

    kernel = rng.random(kernel_size, dtype=xp.float32)

    grad_op = MpiGradient2d(input_size, grid_size, comm)
    conv_op = MpiDftConvolution(input_size, kernel, comm, grid_size)

    shared_comm = SharedCommunicator(
        comm,
        grid_size,
        input_size,
        {"grad": grad_op, "conv": conv_op},
    )

    local_slice = shared_comm.shared_comm.cartslicer.slice_global_buffer_to_tile
    local_X = full_x[local_slice]

    # apply operators with dedicated communicators
    expected_grad = grad_op.forward(local_X)
    expected_conv = conv_op.forward(local_X)

    # apply operators with shared communicator
    shared_comm.update_buffer(local_X)
    actual_grad = shared_comm.apply_operator("grad")
    actual_conv = shared_comm.apply_operator("conv")

    xp.testing.assert_allclose(actual_grad, expected_grad, err_msg="Gradient mismatch")
    xp.testing.assert_allclose(actual_conv, expected_conv, err_msg="Conv mismatch")
