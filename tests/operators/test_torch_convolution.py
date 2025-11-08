from mcmc.operators.mpi_torch_convolution import MpiTorchConvolution

from mpi4py import MPI
import numpy as np

from mcmc.backend import bm, xp
import pytest
import torch


@pytest.fixture
def seed():
    return 1234


@pytest.fixture
def dims():
    return np.array([3, 121, 75], "i")


@pytest.fixture
def kernel_dims():
    return np.array([5, 3], "i")


@pytest.fixture
def padding():
    return tuple([2, 1])


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


@pytest.mark.env("mpi-cpu")
@pytest.mark.env("mpi-gpu")
def test_mpi_convolution(dims, kernel_dims, padding, seed, comm, cmdopt):
    rank = comm.Get_rank()
    if cmdopt == "mpi-gpu":
        bm.set_backend("cupy")
        torch.set_default_device("cuda")

        nb_gpu = xp.cuda.runtime.getDeviceCount()
        gpu_id = rank % nb_gpu

        xp.cuda.runtime.setDevice(gpu_id)
        torch.backends.cudnn.deterministic = True

    comm_size = comm.Get_size()
    rank = comm.Get_rank()
    grid_dims = np.asarray(MPI.Compute_dims(comm_size, 2))
    grid_dims = np.asarray([1, *grid_dims])
    cart_comm = comm.Create_cart(dims=grid_dims)

    rng = xp.random.default_rng(seed)

    X = xp.zeros(tuple(dims), dtype=np.float32)
    kernel = xp.zeros(tuple(kernel_dims), dtype=np.float32)

    if rank == 0:
        X = rng.standard_normal(tuple(dims), dtype=np.float32)
        kernel = rng.random(tuple(kernel_dims), dtype=np.float32)

    cart_comm.Bcast([X, MPI.FLOAT], root=0)
    cart_comm.Bcast([kernel, MPI.FLOAT], root=0)

    conv2d_op = torch.nn.Conv2d(
        in_channels=3, out_channels=3, kernel_size=kernel_dims.tolist(), padding=padding
    )
    conv2d_op.weight.data[:, :] = torch.from_numpy(kernel)

    full_conv = conv2d_op.forward(torch.from_numpy(X))

    torch_conv = MpiTorchConvolution(
        dims, tuple(kernel_dims), padding, comm=comm, grid_size=grid_dims
    )

    local_buffer = np.zeros(
        torch_conv.direct_communicator.cartslicer.facet_size, dtype=np.float32
    )
    local_buffer[torch_conv.direct_communicator.cartslicer.slice_facet_to_tile] = X[
        torch_conv.direct_communicator.cartslicer.slice_global_buffer_to_tile
    ]
    torch_conv.direct_communicator.update_borders(local_buffer)

    torch_conv.forward_no_comm(local_buffer, conv2d_op)
    no_comm_local_Hx = torch_conv.forward_no_comm(
        np.asarray(local_buffer, dtype=np.float32), conv2d_op
    )

    local_Hx = full_conv[
        torch_conv.adjoint_communicator.cartslicer._get_slice_global_buffer_to_tile()
    ]

    assert xp.allclose(
        no_comm_local_Hx,
        np.asarray(local_Hx.detach().numpy(), dtype=np.float32),
        atol=1e-6,
    )  #! float32 accuracy in 1e-6
