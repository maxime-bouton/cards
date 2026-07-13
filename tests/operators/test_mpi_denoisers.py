import numpy as np
import pytest

import cards.backend as xp
from cards.denoisers.distributed_ddfb import DistributedDDFB
from cards.denoisers.distributed_dncnn import DistributedDnCNN
from cards.denoisers.distributed_drunet import DistributedDRUNet
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
def test_distributed_ddfb(seed, input_size, comm, grid_size):
    r"""Verify that the distributed DDFB yields results identical to the serial DDFB."""
    rng = xp.random.default_rng(seed)
    x = rng.random(input_size).astype(xp.float32)

    serial_ddfb = SerialDDFB(
        image_size=input_size,
        n_layers=4,
        n_features=64,
    )

    distributed_ddfb = DistributedDDFB(
        comm,
        grid_size,
        image_size=input_size,
        n_layers=4,
        n_features=64,
    )

    y_serial = serial_ddfb(x, 0.03)[distributed_ddfb.global_to_tile_slice]
    y_mpi = distributed_ddfb(x[distributed_ddfb.global_to_tile_slice], 0.03)

    xp.testing.assert_allclose(y_serial, y_mpi, rtol=1e-5, atol=1e-5)


@pytest.mark.mpi
def test_distributed_dncnn(seed, input_size, comm, grid_size):
    r"""Verify that the distributed DnCNN yields results identical to the serial DnCNN."""
    rng = xp.random.default_rng(seed)
    x = rng.random(input_size).astype(xp.float32)

    serial_dncnn = SerialDnCNN(image_size=input_size)

    distributed_dncnn = DistributedDnCNN(comm, grid_size, image_size=input_size)

    y_serial = serial_dncnn(x, 0.03)[distributed_dncnn.global_to_tile_slice]
    y_mpi = distributed_dncnn(x[distributed_dncnn.global_to_tile_slice], 0.03)

    xp.testing.assert_allclose(y_serial, y_mpi, rtol=1e-5, atol=1e-5)


@pytest.mark.mpi
def test_distributed_drunet(seed, input_size, comm, grid_size):
    r"""Verify that the distributed DRUNet yields results identical to the serial DRUNet.

    Warning
    -------
    DRUNet requires the dimensions of each local tile to be multiples of 8
    (along each spatial axis).
    """
    rng = xp.random.default_rng(seed)
    x = rng.random(input_size).astype(xp.float32)

    serial_drunet = SerialDRUNet(image_size=input_size)

    distributed_drunet = DistributedDRUNet(comm, grid_size, image_size=input_size)

    y_serial = serial_drunet(x, 0.03)[distributed_drunet.global_to_tile_slice]
    y_mpi = distributed_drunet(x[distributed_drunet.global_to_tile_slice], 0.03)

    xp.testing.assert_allclose(y_serial, y_mpi, rtol=1e-5, atol=1e-5)
