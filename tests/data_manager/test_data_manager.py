"""Test for `DataManager` class.

Test the serial/distributed writing/reading of data on/from disk memory.
"""

from os.path import join
from pathlib import Path

import h5py
import numpy as np
import pytest
from mpi4py import MPI

from cards.backend import xp
from cards.communicator.sync_cartesian_communicator import SyncCartesianCommunicator
from cards.data_manager.data_manager import DataManager
from cards.slicer.cartesian_comm_slicer import CartesianCommSlicer


@pytest.fixture
def batch_size():
    return 100


@pytest.mark.serial
@pytest.mark.cpu
def test_save(input_shape: tuple[int, ...], seed: int, tmp_path: Path) -> None:
    """
    Test the saving of data for serial settings on CPU.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(input_shape)
    Y = rng.standard_normal((2, *input_shape))
    Z = rng.standard_normal(input_shape)
    data_manager = DataManager()

    data = {"x": X, "y": Y}

    if tmp_path is not None:
        tmp_path_str = tmp_path.as_posix()
    else:
        tmp_path_str = ""
    filename = join(tmp_path_str, "dummy_save_data.h5")

    with h5py.File(filename, "w") as file:
        data_manager.save_dict(data, file)
        data_manager.save_array(Z, file, "z")

    with h5py.File(filename, "r") as file:
        checkX = np.allclose(file["x"][:], X)
        checkY = np.allclose(file["y"][:], Y)
        checkZ = np.allclose(file["z"][:], Z)

    assert checkX and checkY and checkZ


@pytest.mark.serial
@pytest.mark.cpu
def test_load(tmp_path, input_shape: tuple[int, ...]) -> None:
    """
    Test the loading of data for serial settings on CPU.
    """
    rng = np.random.default_rng(1234)
    X = rng.standard_normal(input_shape)
    Y = rng.standard_normal(np.asarray([2, *input_shape]))
    Z = rng.standard_normal(input_shape)

    if tmp_path is not None:
        tmp_path_str = tmp_path.as_posix()
    else:
        tmp_path_str = ""
    filename = join(tmp_path_str, "dummy_load_data.h5")

    with h5py.File(filename, "w") as file:
        file["x"] = X
        file["y"] = Y
        file["z"] = Z

    data_manager = DataManager()
    with h5py.File(filename, "r") as file:
        data = data_manager.load_h5(file)

    checkX = np.allclose(data["x"], X)
    checkY = np.allclose(data["y"], Y)
    checkZ = np.allclose(data["z"], Z)

    assert checkX and checkY and checkZ


@pytest.mark.serial
def test_save_full_batch(input_shape, tmp_path, device, batch_size):
    """
    Test the saving of full batches for serial settings.
    """
    sizes = {"X": input_shape}

    if tmp_path is not None:
        tmp_path_str = tmp_path.as_posix()
    else:
        tmp_path_str = ""
    filename = join(tmp_path_str, "test_full_batch.h5")

    batched_data = xp.random.standard_normal(size=(batch_size, *input_shape))

    dm = DataManager(batch_size, save_full_batch=True, sizes=sizes)

    dm.full_batch["X"] = batched_data

    with h5py.File(filename, "w") as file:
        dm.save_batch(file, device == "gpu")

    with h5py.File(filename, "r") as file:
        loaded_data = file["batch/X"][:]

    xp.testing.assert_allclose(batched_data, loaded_data)


@pytest.mark.mpi
@pytest.mark.cpu
def test_distributed_save_and_load(
    comm: MPI.Comm,
    input_shape: tuple[int, ...],
    seed: int,
    tmp_path: Path,
) -> None:
    """
    Test the distributed saving and loading of data for CPU settings.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        ss = np.random.SeedSequence(seed)
        # spawn off nworkers child SeedSequences to pass to child processes.
        child_seed = ss.spawn(size)
    else:
        child_seed = None

    local_seed = comm.scatter(child_seed, root=0)
    rng = np.random.default_rng(local_seed)

    grid_dims = np.asarray([1, *MPI.Compute_dims(size, 2)], dtype=int)

    sync_comm = SyncCartesianCommunicator(
        comm,
        grid_dims,
        np.asarray(input_shape),
        np.zeros(len(input_shape)),
        np.zeros(len(input_shape)),
    )
    slice_facet_to_tile = sync_comm.cartslicer._get_slice_global_buffer_to_tile()

    local_dim = sync_comm.cartslicer.tile_size
    x = rng.standard_normal(size=local_dim)

    data_manager = DataManager()
    filename = ""

    if rank == 0:
        if tmp_path is not None:
            tmp_path_str = tmp_path.as_posix()
        else:
            tmp_path_str = ""
        filename = join(tmp_path_str, "distributed_save_data.h5")

    filename = comm.bcast(filename, 0)

    data = {"x": x}
    slices = {"x": slice_facet_to_tile}
    global_sizes = {"x": input_shape}
    local_sizes = {"x": local_dim}

    with h5py.File(filename, "w", driver="mpio", comm=comm) as file:
        data_manager.save_dict(data, file, global_sizes, slices)

    comm.Barrier()

    with h5py.File(filename, "r", driver="mpio", comm=comm) as file:
        data = data_manager.load_h5(file, local_sizes, slices)

    check = np.allclose(x, data["x"])

    all_check = comm.reduce(check, op=MPI.PROD, root=0)

    if rank == 0:
        assert all_check


@pytest.mark.mpi
def test_mpi_save_full_batch(comm, input_shape, tmp_path, device, batch_size):
    """
    Test the distributed saving of full batches for MPI settings.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    grid_dims = np.asarray([1, *MPI.Compute_dims(size, 2)], dtype=int)
    mpi_cart_comm = comm.Create_cart(grid_dims)
    ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))

    filename = ""
    if rank == 0:
        if tmp_path is not None:
            tmp_path_str = tmp_path.as_posix()
        else:
            tmp_path_str = ""
        filename = join(tmp_path_str, "test_mpi_full_batch.h5")

    filename = comm.bcast(filename, 0)

    slicer = CartesianCommSlicer(
        ranknd,
        grid_dims,
        input_shape,
        np.zeros(len(input_shape)),
        np.zeros(len(input_shape)),
    )

    local_slice = {}
    local_slice["X"] = slicer._get_slice_global_buffer_to_tile()
    local_size = {}
    local_size["X"] = slicer.tile_size
    global_dims = {}
    global_dims["X"] = input_shape

    dm = DataManager(
        batch_size,
        save_full_batch=True,
        sizes=local_size,
        global_sizes=global_dims,
        local_slices=local_slice,
    )

    data = xp.zeros((batch_size, *input_shape))
    if rank == 0:
        data = xp.random.standard_normal(size=data.shape)

    data = comm.bcast(data, 0)
    data_slice = np.s_[slice(None), *slicer._get_slice_global_buffer_to_tile()]
    local_data = data[data_slice].copy()

    dm.full_batch["X"] = local_data

    with h5py.File(filename, "w", driver="mpio", comm=comm) as file:
        dm.save_batch(file, device == "gpu")

    loaded_data = np.zeros(local_data.shape)

    with h5py.File(filename, "r", driver="mpio", comm=comm) as file:
        file["batch/X"].read_direct(loaded_data, source_sel=data_slice)

    local_check = xp.isclose(xp.asarray(local_data), loaded_data).all()

    global_check = False
    global_check = comm.reduce(local_check, MPI.PROD, root=0)

    if rank == 0:
        assert global_check


@pytest.mark.serial
@pytest.mark.cpu
def test_save_and_load_rng(tmp_path, input_shape, seed, seed2) -> None:
    """
    Test the saving and loading of the state of a random number generator for serial settings on CPU.
    """
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed2)
    n_trials = 10

    for i in range(n_trials):
        rng.standard_normal(input_shape)

    data_manager = DataManager()
    if tmp_path is not None:
        tmp_path_str = tmp_path.as_posix()
    else:
        tmp_path_str = ""
    filename = join(tmp_path_str, "test_rng.h5")

    with h5py.File(filename, "w") as file:
        data_manager.save_rng(rng, file)

    with h5py.File(filename, "r") as file:
        data_manager.load_rng(rng2, file)

    check = np.zeros(n_trials, dtype=bool)

    for i in range(n_trials):
        check[i] = np.allclose(
            rng.standard_normal(input_shape),
            rng2.standard_normal(input_shape),
        )

    assert check.all()
