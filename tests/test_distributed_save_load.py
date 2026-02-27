r"""Test the paralell writting/reading of data on/from disk memory."""

from os.path import join

import h5py
import numpy as np
import pytest
from mpi4py import MPI

from mcmc.communicator.sync_cartesian_communicator import SyncCartesianCommunicator
from mcmc.data_manager.data_manager import DataManager

pytestmark = pytest.mark.mpi

# TODO: add example with pytorch


@pytest.fixture
def dims():
    return np.asarray([100, 50], dtype=int)


@pytest.fixture
def seed():
    return 1234


@pytest.mark.numpy
def test_distributed_save(tmp_path, dims, seed):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        ss = np.random.SeedSequence(seed)
        # spawn off nworkers child SeedSequences to pass to child processes.
        child_seed = np.array(ss.spawn(size))
    else:
        child_seed = None

    local_seed = comm.scatter(child_seed, root=0)
    rng = np.random.default_rng(local_seed)

    grid_dims = np.asarray(MPI.Compute_dims(comm.Get_size(), 2), dtype=int)

    sync_comm = SyncCartesianCommunicator(
        MPI.COMM_WORLD, grid_dims, dims, np.asarray([0, 0]), np.asarray([0, 0])
    )
    slice_facet_to_tile = sync_comm.cartslicer._get_slice_global_buffer_to_tile()

    local_dim = sync_comm.cartslicer.tile_size
    X = rng.standard_normal(size=local_dim, dtype=np.float64)

    data_manager = DataManager()

    if tmp_path is not None:
        tmp_path_str = tmp_path.as_posix()
    else:
        tmp_path_str = ""
    filename = join(tmp_path_str, "distributed_save_data.h5")

    data = {"x": X}
    slices = {"x": slice_facet_to_tile}
    global_sizes = {"x": dims}

    with h5py.File(filename, "w", driver="mpio", comm=MPI.COMM_WORLD) as file:
        data_manager.save_dict(data, file, global_sizes, slices)

    comm.Barrier()

    with h5py.File(filename, "r", driver="mpio", comm=MPI.COMM_WORLD) as file:
        data = data_manager.load_h5(file, slices)

    check = np.allclose(X, data["x"])

    all_check = False
    all_check = comm.reduce(check, op=MPI.PROD, root=0)

    if rank == 0:
        assert all_check


if __name__ == "__main__":
    dims = np.asarray([100, 50], dtype=int)
    tmp_path = None
    seed = 1234

    test_distributed_save(tmp_path, dims, seed)
