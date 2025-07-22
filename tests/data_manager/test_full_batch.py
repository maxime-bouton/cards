import numpy as np
import cupy as cp
from os.path import join
import pytest
from mcmc.data_manager.data_manager import DataManager
from mcmc.backend import set_backend
import h5py
from mpi4py import MPI
from mcmc.slicer.cartesian_comm_slicer import CartesianCommSlicer


@pytest.fixture
def dims():
    return np.asarray([100, 50], dtype=int)


@pytest.fixture
def seed():
    return 1234


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


@pytest.mark.env("serial-cpu")
@pytest.mark.env("serial-gpu")
def test_save_full_batch(dims, cmdopt, tmp_path):
    batch_size = 100
    sizes = {}
    sizes["X"] = dims

    if tmp_path is not None:
        tmp_path_str = tmp_path.as_posix()
    else:
        tmp_path_str = ""
    filename = join(tmp_path_str, "test_full_batch.h5")

    from_gpu = False
    if cmdopt == "serial-gpu":
        set_backend("cupy")
        from_gpu = True

    from mcmc.backend import xp

    data = xp.random.standard_normal(size=(batch_size, *dims))

    dm = DataManager(batch_size, True, sizes)

    dm.full_batch["X"] = data

    with h5py.File(filename, "w") as file:
        dm.save_batch(file, from_gpu)

    with h5py.File(filename, "r") as file:
        loaded_data = file["batch/X"][:]

    assert xp.isclose(data, loaded_data).all()


@pytest.mark.env("mpi-cpu")
@pytest.mark.env("mpi-gpu")
def test_mpi_save_full_batch(comm, dims, cmdopt, tmp_path):
    batch_size = 100
    rank = comm.Get_rank()
    grid_dims = np.asarray(MPI.Compute_dims(comm.Get_size(), 2), dtype=int)
    mpi_cart_comm = comm.Create_cart(grid_dims)
    ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))

    from_gpu = False
    if cmdopt == "mpi-gpu":
        set_backend("cupy")
        from_gpu = True
        gpu_id = rank % cp.cuda.runtime.getDeviceCount()
        cp.cuda.runtime.setDevice(gpu_id)
    from mcmc.backend import xp

    filename = ""
    if rank == 0:
        if tmp_path is not None:
            tmp_path_str = tmp_path.as_posix()
        else:
            tmp_path_str = ""
        filename = join(tmp_path_str, "test_mpi_full_batch.h5")

    filename = comm.bcast(filename, 0)

    slicer = CartesianCommSlicer(
        ranknd, grid_dims, dims, np.asarray([0, 0]), np.asarray([0, 0])
    )
    local_slice = {}
    local_slice["X"] = slicer._get_slice_global_buffer_to_tile()
    local_size = {}
    local_size["X"] = slicer.tile_size
    global_dims = {}
    global_dims["X"] = dims

    dm = DataManager(batch_size, True, local_size, global_dims, local_slice)

    data = xp.zeros((batch_size, *dims))
    if rank == 0:
        data = xp.random.standard_normal(size=data.shape)

    data = comm.bcast(data, 0)
    data_slice = np.s_[slice(None), *slicer._get_slice_global_buffer_to_tile()]
    local_data = data[data_slice].copy()

    dm.full_batch["X"] = local_data

    with h5py.File(filename, "w", driver="mpio", comm=comm) as file:
        dm.save_batch(file, from_gpu)

    loaded_data = np.zeros(local_data.shape)

    with h5py.File(filename, "r", driver="mpio", comm=comm) as file:
        file["batch/X"].read_direct(loaded_data, source_sel=data_slice)

    local_check = xp.isclose(xp.asarray(local_data), loaded_data).all()

    global_check = False
    global_check = comm.reduce(local_check, MPI.PROD, root=0)

    if rank == 0:
        assert global_check
