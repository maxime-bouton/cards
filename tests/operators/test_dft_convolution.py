import numpy as np
import pytest
from mpi4py import MPI

import cards.backend as xp
from cards.operators.dft_convolution import DftConvolution
from cards.operators.distributed_dft_convolution import DistributedDftConvolution
from cards.utils.utils import expand_shape_left

# FIXME: cleanse distributed test, avoid copying full array on all workers


@pytest.fixture
def kernel_size(input_shape) -> np.ndarray:
    return np.array(expand_shape_left((5, 3), ndim=len(input_shape)))


@pytest.fixture
def input_size(input_shape) -> np.ndarray:
    return np.array(input_shape)


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


@pytest.mark.serial
def test_adjoint(seed, input_shape, input_size, kernel_size):
    """
    Test the adjoint property of the DFT convolution operator in serial setting.
    """
    data_size = input_size + kernel_size - 1
    data_shape = (*data_size,)
    rng = xp.random.default_rng(seed)
    X = rng.random(input_size)
    Y = rng.random(data_size)
    kernel = rng.random(kernel_size)

    conv = DftConvolution(input_shape, data_shape, kernel)

    Hx = conv.forward(X)
    Hy = conv.adjoint(Y)

    Hxy = xp.sum(Hx * Y)
    xHy = xp.sum(X * Hy)

    xp.testing.assert_allclose(Hxy, xHy, atol=1e-10)


@pytest.mark.mpi
def test_adjoint_mpi(
    seed, input_shape, input_size, kernel_size, comm, rank, grid_shape
):
    """
    Test the adjoint property of the DFT convolution operator in distributed settings.
    """
    output_size = input_size + kernel_size - 1

    rng = xp.random.default_rng(seed)
    X = rng.random(input_size)
    Y = rng.random(output_size)
    kernel = rng.random(kernel_size)

    convolution_handler = DistributedDftConvolution(
        input_shape,
        grid_shape,
        comm,
        kernel,
    )

    local_X = X[
        convolution_handler.direct_communicator.cartslicer.slice_global_buffer_to_tile
    ]
    local_Y = Y[
        convolution_handler.adjoint_communicator.cartslicer.slice_global_buffer_to_tile
    ]

    local_Hx = convolution_handler.forward(local_X)
    local_Hy = convolution_handler.adjoint(local_Y)

    local_Hxy = xp.sum(local_Hx * local_Y)
    local_xHy = xp.sum(local_X * local_Hy)

    Hxy = 0
    xHy = 0

    Hxy = comm.allreduce(local_Hxy, MPI.SUM)
    xHy = comm.allreduce(local_xHy, MPI.SUM)

    xp.testing.assert_allclose(Hxy, xHy)
