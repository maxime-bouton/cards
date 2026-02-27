import cupy as cp
import numpy as np
import pytest
import torch
from mpi4py import MPI

from mcmc.backend import bm
from mcmc.denoisers.mpi_ddfb import MpiDDFB
from mcmc.denoisers.mpi_dncnn import MpiDnCNN
from mcmc.denoisers.mpi_drunet import MpiDRUNet
from mcmc.denoisers.serial_ddfb import SerialDDFB
from mcmc.denoisers.serial_dncnn import SerialDnCNN
from mcmc.denoisers.serial_drunet import SerialDRUNet
from mcmc.utils.utils_img import load_img


@pytest.fixture
def seed():
    return 1234


@pytest.fixture
def dims():
    return np.array([3, 128, 128], "i")


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


@pytest.fixture
def grid_size(comm):
    comm_size = comm.Get_size()
    return np.asarray([1] + MPI.Compute_dims(comm_size, 2))


@pytest.mark.env("mpi-gpu")
def test_mpi_ddfb(seed, dims, comm, grid_size):
    bm.set_backend("cupy")
    torch.set_default_device("cuda")
    torch.backends.cudnn.deterministic = True

    rng = cp.random.default_rng(seed)
    x = load_img("data/180.h5", key="x") + 0.03 * rng.standard_normal(dims).astype(
        np.float32
    )

    serial_ddfb = SerialDDFB(
        image_size=dims,
        n_layers=4,
        n_features=64,
    )

    mpi_ddfb = MpiDDFB(
        comm,
        grid_size,
        image_size=dims,
        n_layers=4,
        n_features=64,
    )

    y_serial = serial_ddfb(x, 0.03)[mpi_ddfb.global_to_tile_slice]
    y_mpi = mpi_ddfb(x[mpi_ddfb.global_to_tile_slice], 0.03)

    cp.testing.assert_allclose(y_serial, y_mpi)


@pytest.mark.env("mpi-gpu")
def test_mpi_dncnn(seed, dims, comm, grid_size):
    bm.set_backend("cupy")
    torch.set_default_device("cuda")
    torch.backends.cudnn.deterministic = True

    rng = cp.random.default_rng(seed)
    x = rng.random(dims).astype(cp.float32)

    serial_dncnn = SerialDnCNN(image_size=dims)

    mpi_dncnn = MpiDnCNN(comm, grid_size, image_size=dims)

    y_serial = serial_dncnn(x, 0.03)[mpi_dncnn.global_to_tile_slice]
    y_mpi = mpi_dncnn(x[mpi_dncnn.global_to_tile_slice], 0.03)

    cp.testing.assert_allclose(y_serial, y_mpi)


@pytest.mark.env("mpi-gpu")
def test_mpi_drunet(seed, dims, comm, grid_size):
    bm.set_backend("cupy")
    torch.set_default_device("cuda")
    torch.backends.cudnn.deterministic = True

    rng = cp.random.default_rng(seed)
    x = rng.random(dims).astype(cp.float32)

    serial_drunet = SerialDRUNet(image_size=dims)

    mpi_drunet = MpiDRUNet(comm, grid_size, image_size=dims)

    y_serial = serial_drunet(x, 0.03)[mpi_drunet.global_to_tile_slice]
    y_mpi = mpi_drunet(x[mpi_drunet.global_to_tile_slice], 0.03)

    cp.testing.assert_allclose(y_serial, y_mpi)


# mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m mpi4py -m pytest test_dft_convolution.py -C mpi-gpu
