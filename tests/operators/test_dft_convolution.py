r"""Testing implementation consistency and error messages for DFT-based convolution operators."""

import numpy as np
import pytest
from mpi4py import MPI

import cards.backend as xp
from cards.operators.dft_convolution import DftConvolution
from cards.operators.distributed_dft_convolution import DistributedDftConvolution
from cards.utils.utils import expand_shape_left

# FIXME: improve rng management for mpi tests (use create_rng, separate cpu and gpu configs)


@pytest.fixture
def kernel_size(input_shape) -> np.ndarray:
    return np.array(expand_shape_left((5, 3), ndim=len(input_shape)))


@pytest.mark.serial
def test_dftconv_error_messages(input_size, kernel_size):

    data_size = input_size + kernel_size - 1

    # error complex-valued kernels
    with pytest.raises(TypeError) as excinfo:
        DftConvolution((*input_size,), (*data_size,), (1 + 1j) * xp.ones(kernel_size))
    assert "only real-valued kernel supported" in str(excinfo.value)

    # error kernel size (number of axes incompatiable with input shape)
    with pytest.raises(ValueError) as excinfo:
        DftConvolution((*input_size,), (*data_size,), xp.ones(kernel_size[-1:]))
    assert "kernel should have ndims = len(image_size) dimensions" in str(excinfo.value)

    # error input and output shape (inconsistent number of axes between input and output)
    with pytest.raises(ValueError) as excinfo:
        DftConvolution((*input_size,), (*data_size[-2:],), xp.ones(kernel_size))
    assert "image_shape and data_shape must have the same number of elements" in str(
        excinfo.value
    )


@pytest.mark.serial
def test_adjoint(seed, input_shape, input_size, kernel_size):
    """Serial test to check the implementation of the adjoint operator is consistent with the direct operator."""
    rng = xp.random.default_rng(seed)

    kernel = rng.random(kernel_size)
    x = rng.random(input_size)
    data_size = input_size + kernel_size - 1
    y = rng.random(data_size)

    conv = DftConvolution(input_shape, (*data_size,), kernel)

    Hx = conv.forward(x)
    Hadj_y = conv.adjoint(y)

    Hxy = xp.sum(Hx * y)
    xHadj_y = xp.sum(x * Hadj_y)

    xp.testing.assert_allclose(Hxy, xHadj_y)


@pytest.mark.mpi
def test_dftconv_error_messages_mpi(input_shape, kernel_size, comm, grid_shape):

    # error complex-valued kernels
    with pytest.raises(TypeError) as excinfo:
        DistributedDftConvolution(
            input_shape,
            grid_shape,
            comm,
            (1 + 1j) * xp.ones(kernel_size),
        )
    assert "only real-valued kernel supported" in str(excinfo.value)

    # error kernel size (number of axes incompatiable with input shape)
    with pytest.raises(ValueError) as excinfo:
        DistributedDftConvolution(
            input_shape,
            grid_shape,
            comm,
            (1 + 1j) * xp.ones(kernel_size[-2:]),
        )
    assert "kernel should have ndims = len(image_size) dimensions" in str(excinfo.value)


@pytest.mark.mpi
def test_adjoint_mpi(seed, input_shape, kernel_size, comm, rank, grid_shape):
    """Distributed test to check the implementation of the adjoint operator is consistent with the direct operator."""
    kernel_rng = xp.random.default_rng(seed)
    kernel = kernel_rng.random(kernel_size)
    convolution_handler = DistributedDftConvolution(
        input_shape,
        grid_shape,
        comm,
        kernel,
    )

    # draw local image tile and observation tile
    rng = xp.random.default_rng(seed)
    local_x = rng.random(convolution_handler.direct_communicator.cartslicer.tile_size)
    local_y = rng.random(convolution_handler.adjoint_communicator.cartslicer.tile_size)

    local_Hx = convolution_handler.forward(local_x)
    local_Hadj_y = convolution_handler.adjoint(local_y)

    local_Hxy = xp.sum(local_Hx * local_y)
    local_xHadj_y = xp.sum(local_x * local_Hadj_y)

    Hxy = comm.allreduce(local_Hxy, MPI.SUM)
    xHadj_y = comm.allreduce(local_xHadj_y, MPI.SUM)

    xp.testing.assert_allclose(Hxy, xHadj_y)
