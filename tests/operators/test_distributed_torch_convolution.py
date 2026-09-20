r"""Testing implementation consistency for tirch-based convolution operators."""

import pytest
import torch
from mpi4py import MPI

import cards.backend as xp
from cards.operators.distributed_torch_convolution import DistributedTorchConvolution
from cards.utils.utils import torch2xp, xp2torch

# FIXME: avoid full arrays on all workers
# TODO: revise the way rng is handled in MPI test


@pytest.fixture
def kernel_dims():
    return (5, 3)


@pytest.fixture
def padding():
    return (2, 1)


@pytest.mark.mpi
def test_mpi_torch_conv(input_shape, kernel_dims, padding, seed, comm, comm_size):
    """Test torch-based convolution operator produces the same in serial and distributed configurations."""
    grid_dims = (1, comm_size, 1)
    Cin = input_shape[0]
    rng = xp.random.default_rng(seed)

    conv = DistributedTorchConvolution(
        input_shape,
        kernel_dims,
        padding,
        comm,
        grid_dims,
    )

    # generate random kernel for convolution
    kernel = xp2torch(
        rng.random((Cin, Cin) + kernel_dims).astype(xp.float32),
        add_batch=False,
    )

    # define forward serial convolution operator
    torch_conv = torch.nn.Conv2d(
        in_channels=Cin,
        out_channels=Cin,
        kernel_size=kernel_dims,
        padding=padding,
        bias=False,
    )
    torch_conv.weight.data = kernel

    # * check consistency between full and distributed forward operator
    # generate full input x and global convolution
    full_x = rng.random(input_shape).astype(xp.float32)
    full_conv = torch2xp(torch_conv(xp2torch(full_x)))
    Hx_serial = full_conv[
        conv.adjoint_communicator.cartslicer.slice_global_buffer_to_tile
    ]

    # get local input tile and MPI convolution
    local_x = full_x[conv.direct_communicator.cartslicer.slice_global_buffer_to_tile]
    Hx_mpi = conv.forward(local_x, torch_conv)

    xp.testing.assert_allclose(Hx_serial, Hx_mpi, atol=1e-7)

    # * check consistency between forward and adjoint operators
    # define serial adjoint convolution operator
    torch_adjoint_conv = torch.nn.ConvTranspose2d(
        Cin, Cin, kernel_dims, padding=padding, bias=False
    )
    torch_adjoint_conv.weight.data = kernel

    # generate full observaiton y and global adjoint convolution
    full_y = rng.random((*conv.data_shape,)).astype(xp.float32)
    full_adj_conv = torch2xp(torch_adjoint_conv(xp2torch(full_y)))
    Hadj_y_serial = full_adj_conv[
        conv.direct_communicator.cartslicer.slice_global_buffer_to_tile
    ]

    # get local observation tile and MPI adjoint convolution
    local_y = full_y[conv.adjoint_communicator.cartslicer.slice_global_buffer_to_tile]
    Hadj_y_mpi = conv.adjoint(local_y, torch_adjoint_conv)

    # local consistency (output of adjoint operator)
    xp.testing.assert_allclose(Hadj_y_serial, Hadj_y_mpi, atol=1e-7)

    # global consistency
    local_sp1 = xp.sum(Hx_mpi * local_y)
    local_sp2 = xp.sum(local_x * Hadj_y_mpi)

    sp1 = comm.allreduce(local_sp1, MPI.SUM)
    sp2 = comm.allreduce(local_sp2, MPI.SUM)

    xp.testing.assert_allclose(sp1, sp2, atol=1e-2)
