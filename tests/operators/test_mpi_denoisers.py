import numpy as np
import pytest

import cards.backend as xp
from cards.denoisers.mpi_ddfb import MpiDDFB
from cards.denoisers.mpi_dncnn import MpiDnCNN
from cards.denoisers.mpi_drunet import MpiDRUNet
from cards.denoisers.serial_ddfb import SerialDDFB
from cards.denoisers.serial_dncnn import SerialDnCNN
from cards.denoisers.serial_drunet import SerialDRUNet

# TODO: add test with gray images (n_channels = 1), missing DDFB weights with nch=1


# NOTE: only first spatial axis is partitioned because way too slow when both axes are.
@pytest.fixture
def grid_size(comm_size: int) -> np.ndarray:
    return np.asarray([1, comm_size, 1])


# NOTE: only one input shape configuration: DRUNet requires the
# dimensions of each local tile to be multiples of 8 (on each spatial axis).
@pytest.fixture(params=[(3, 128, 128)])
def input_shape(request):
    return request.param


@pytest.fixture
def input_size(input_shape) -> np.ndarray:
    return np.array(input_shape)


@pytest.mark.mpi
def test_mpi_ddfb(seed, input_size, comm, grid_size):
    r"""Verify that the distributed DDFB yields results identical to the serial DDFB."""
    rng = xp.random.default_rng(seed)
    x = rng.random(input_size).astype(xp.float32)

    serial_ddfb = SerialDDFB(
        image_size=input_size,
        n_layers=4,
        n_features=64,
    )

    mpi_ddfb = MpiDDFB(
        comm,
        grid_size,
        image_size=input_size,
        n_layers=4,
        n_features=64,
    )

    y_serial = serial_ddfb(x, 0.03)[mpi_ddfb.global_to_tile_slice]
    y_mpi = mpi_ddfb(x[mpi_ddfb.global_to_tile_slice], 0.03)

    xp.testing.assert_allclose(y_serial, y_mpi, rtol=1e-5, atol=1e-5)


@pytest.mark.mpi
def test_mpi_dncnn(seed, input_size, comm, grid_size):
    r"""Verify that the distributed DnCNN yields results identical to the serial DnCNN."""
    rng = xp.random.default_rng(seed)
    x = rng.random(input_size).astype(xp.float32)

    serial_dncnn = SerialDnCNN(image_size=input_size)

    mpi_dncnn = MpiDnCNN(comm, grid_size, image_size=input_size)

    y_serial = serial_dncnn(x, 0.03)[mpi_dncnn.global_to_tile_slice]
    y_mpi = mpi_dncnn(x[mpi_dncnn.global_to_tile_slice], 0.03)

    xp.testing.assert_allclose(y_serial, y_mpi, rtol=1e-5, atol=1e-5)


@pytest.mark.mpi
def test_mpi_drunet(seed, input_size, comm, grid_size):
    r"""Verify that the distributed DRUNet yields results identical to the serial DRUNet.

    Warning
    -------
    DRUNet requires the dimensions of each local tile to be multiples of 8
    (along each spatial axis).
    """
    rng = xp.random.default_rng(seed)
    x = rng.random(input_size).astype(xp.float32)

    serial_drunet = SerialDRUNet(image_size=input_size)

    mpi_drunet = MpiDRUNet(comm, grid_size, image_size=input_size)

    y_serial = serial_drunet(x, 0.03)[mpi_drunet.global_to_tile_slice]
    y_mpi = mpi_drunet(x[mpi_drunet.global_to_tile_slice], 0.03)

    xp.testing.assert_allclose(y_serial, y_mpi, rtol=1e-5, atol=1e-5)
