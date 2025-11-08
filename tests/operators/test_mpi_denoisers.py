import numpy as np
import pytest
import torch
from mpi4py import MPI

from mcmc.backend import bm, xp
from mcmc.denoisers.mpi_ddfb import MpiDDFB
from mcmc.denoisers.mpi_dncnn import MpiDnCNN
from mcmc.denoisers.serial_ddfb import SerialDDFB
from mcmc.denoisers.serial_dncnn import SerialDnCNN


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
@pytest.mark.env("mpi-cpu")
def test_mpi_ddfb(seed, dims, comm, grid_size, cmdopt):
    rank = comm.Get_rank()
    if cmdopt == "mpi-gpu":
        bm.set_backend("cupy")
        torch.set_default_device("cuda")

        nb_gpu = xp.cuda.runtime.getDeviceCount()
        gpu_id = rank % nb_gpu

        xp.cuda.runtime.setDevice(gpu_id)

        torch.backends.cudnn.deterministic = True

    rng = xp.random.default_rng(seed)
    x = rng.random(dims).astype(xp.float32)

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

    assert xp.allclose(y_serial, y_mpi)


@pytest.mark.env("mpi-gpu")
@pytest.mark.env("mpi-cpu")
def test_mpi_dncnn(seed, dims, comm, grid_size, cmdopt):
    if cmdopt == "mpi-gpu":
        bm.set_backend("cupy")
        torch.set_default_device("cuda")
        torch.backends.cudnn.deterministic = True

    rng = xp.random.default_rng(seed)
    x = rng.random(dims).astype(xp.float32)

    serial_dncnn = SerialDnCNN(image_size=dims)

    mpi_dncnn = MpiDnCNN(comm, grid_size, image_size=dims)

    y_serial = serial_dncnn(x, 0.03)[mpi_dncnn.global_to_tile_slice]
    y_mpi = mpi_dncnn(x[mpi_dncnn.global_to_tile_slice], 0.03)

    assert xp.allclose(y_serial, y_mpi, atol=1e-6)


@pytest.mark.env("mpi-cpu")
@pytest.mark.env("mpi-gpu")
def test_ddfb_no_comm(dims, grid_size, seed, comm, cmdopt):
    if cmdopt == "mpi-gpu":
        bm.set_backend("cupy")
        torch.set_default_device("cuda")
        torch.backends.cudnn.deterministic = True

    rank = comm.Get_rank()
    cart_comm = comm.Create_cart(dims=grid_size)

    rng = np.random.default_rng(seed)

    X = xp.zeros(dims, dtype=np.float32)

    if rank == 0:
        X = rng.standard_normal(dims, dtype=np.float32)

    cart_comm.Bcast([X, MPI.FLOAT], root=0)

    mpi_ddfb = MpiDDFB(
        comm,
        grid_size,
        image_size=dims,
        n_layers=4,
        n_features=64,
    )

    y_mpi = mpi_ddfb(X[mpi_ddfb.global_to_tile_slice], 0.03)

    local_buffer = xp.zeros(
        mpi_ddfb.mpi_conv.direct_communicator.cartslicer.facet_size, dtype=np.float32
    )

    local_buffer[
        mpi_ddfb.mpi_conv.direct_communicator.cartslicer.slice_facet_to_tile
    ] = X[mpi_ddfb.mpi_conv.direct_communicator.cartslicer.slice_global_buffer_to_tile]

    mpi_ddfb.mpi_conv.direct_communicator.update_borders(local_buffer)

    y_no_comm = mpi_ddfb.forward_no_comm(local_buffer, 0.03)

    local_check = xp.allclose(y_mpi, y_no_comm)

    global_check = comm.reduce(local_check, MPI.PROD, root=0)

    if rank == 0:
        assert global_check


@pytest.mark.env("mpi-cpu")
@pytest.mark.env("mpi-gpu")
def test_dncnn_no_comm(dims, grid_size, seed, comm, cmdopt):
    rank = comm.Get_rank()
    cart_comm = comm.Create_cart(dims=grid_size)

    if cmdopt == "mpi-gpu":
        bm.set_backend("cupy")
        torch.set_default_device("cuda")
        nb_gpu = xp.cuda.runtime.getDeviceCount()
        gpu_id = rank % nb_gpu

        xp.cuda.runtime.setDevice(gpu_id)
        torch.backends.cudnn.deterministic = True

    rng = np.random.default_rng(seed)

    X = xp.zeros(dims, dtype=np.float32)

    if rank == 0:
        X = rng.standard_normal(dims, dtype=np.float32)

    cart_comm.Bcast([X, MPI.FLOAT], root=0)

    mpi_dncnn = MpiDnCNN(
        comm,
        grid_size,
        image_size=dims,
    )

    y_mpi = mpi_dncnn(X[mpi_dncnn.global_to_tile_slice], 0.03)

    local_buffer = xp.zeros(
        mpi_dncnn.edge_mpi_conv.direct_communicator.cartslicer.facet_size,
        dtype=np.float32,
    )

    local_buffer[
        mpi_dncnn.edge_mpi_conv.direct_communicator.cartslicer.slice_facet_to_tile
    ] = X[
        mpi_dncnn.edge_mpi_conv.direct_communicator.cartslicer.slice_global_buffer_to_tile
    ]

    mpi_dncnn.edge_mpi_conv.direct_communicator.update_borders(local_buffer)

    y_no_comm = mpi_dncnn.forward_no_comm(local_buffer, 0.03)

    local_check = xp.allclose(y_mpi, y_no_comm)

    global_check = comm.reduce(local_check, MPI.PROD, root=0)

    if rank == 0:
        assert global_check


# mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m mpi4py -m pytest test_dft_convolution.py -C mpi-gpu
