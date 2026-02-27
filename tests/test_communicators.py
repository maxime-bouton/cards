import cupy as cp
import numpy as np
import pytest
from mpi4py import MPI

from mcmc.communicator.sync_cartesian_communicator import SyncCartesianCommunicator


@pytest.fixture
def seed():
    return 1234


@pytest.fixture
def dims():
    return np.array([3, 5, 5], "i")


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


@pytest.fixture
def grid_size(comm):
    comm_size = comm.Get_size()
    return np.asarray([1] + MPI.Compute_dims(comm_size, 2))


@pytest.fixture
def dtype():
    return np.float32


@pytest.mark.parametrize("backward", [True, False])
@pytest.mark.env("mpi-gpu")
def test_communication_2d(seed, comm, dtype, backward):
    """Test communication of chunks of a 2d array axis, each along axis of the Cartesian grid the axis."""

    rank = comm.Get_rank()
    size = comm.Get_size()

    N = np.array([3, 16, 16], dtype="i")
    M = np.array([1, 3, 3], dtype="i")
    overlap_send = M - 1
    overlap_recv = overlap_send.copy()

    ndims = N.size
    grid_size = np.array(MPI.Compute_dims(size, ndims), dtype="i")

    communicator = SyncCartesianCommunicator(
        comm,
        grid_size,
        N,
        overlap_send,
        overlap_recv,
        dtype=dtype,
        backward=backward,
    )

    rng = np.random.default_rng(seed)
    local_random = rng.random(dtype=dtype)

    facet = cp.full(
        communicator.cartslicer.facet_size,
        rank + local_random,
        dtype=dtype,
    )

    communicator.update_borders(facet)

    # checking consistency of the results
    nregions = communicator.ndims * (communicator.ndims - 1) + 1
    s0 = cp.zeros(nregions, dtype=dtype)
    for k in range(nregions):
        s0[k] = cp.sum(facet[communicator.cartslicer.slice_async_recv[k]])

    expected_sum = cp.sum(
        cp.full(
            communicator.cartslicer.tile_size,
            rank + local_random,
            dtype=dtype,
        )
    ) + cp.sum(s0)

    cp.testing.assert_allclose(cp.sum(facet), expected_sum)

    comm.Barrier()
    communicator.remove()
    comm.Barrier()
