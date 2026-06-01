import numpy as np
import pytest
import torch

from cards.backend import xp
from cards.operators.mpi_torch_convolution import MpiTorchConvolution
from cards.utils.utils import torch2xp, xp2torch


@pytest.fixture
def kernel_dims():
    return (5, 3)


@pytest.fixture
def padding():
    return (2, 1)


# FIXME: currently failing with 9 workers, to investigate
@pytest.mark.mpi
def test_mpi_torch_conv(input_shape, kernel_dims, padding, seed, comm, comm_size):
    """
    Test that the MPI Torch convolution operator produces the same result as the
    corresponding tile of the serial version.
    """
    # grid_dims = np.asarray([1, *MPI.Compute_dims(comm_size, 2)])
    grid_dims = np.asarray([1, comm_size, 1])
    Cin = input_shape[0]
    rng = xp.random.default_rng(seed)

    # define MPI convolution operator
    conv = MpiTorchConvolution(
        np.asarray(input_shape),
        kernel_dims,
        padding,
        comm=comm,
        grid_size=grid_dims,
    )

    # generate random kernel for convolution (same for serial and MPI)
    kernel = xp2torch(
        rng.random((Cin, Cin) + kernel_dims).astype(xp.float32),
        add_batch=False,
    )

    # define serial convolution operator
    torch_conv = torch.nn.Conv2d(
        in_channels=Cin,
        out_channels=Cin,
        kernel_size=kernel_dims,
        padding=padding,
        bias=False,
    )
    torch_conv.weight.data = kernel

    # generate full input x
    full_x = rng.random(input_shape).astype(xp.float32)

    # get local x tile for MPI convolution
    local_x = full_x[conv.direct_communicator.cartslicer.slice_global_buffer_to_tile]

    # compute serial convolution and extract local output tile
    full_conv = torch2xp(torch_conv(xp2torch(full_x)))
    serial = full_conv[conv.adjoint_communicator.cartslicer.slice_global_buffer_to_tile]

    # compute MPI convolution
    mpi = conv.forward(local_x, torch_conv)

    xp.testing.assert_allclose(serial, mpi, atol=1e-7)
