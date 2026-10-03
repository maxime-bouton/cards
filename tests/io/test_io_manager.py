"""Test for `IOManager` class.

Test the serial/distributed writing/reading of data on/from disk memory.
"""

from os.path import join
from pathlib import Path

import h5py
import numpy as np
import pytest

import cards.backend as xp
from cards.communicators.sync_cartesian_communicator import SyncCartesianCommunicator
from cards.core.execution_context import ExecutionContext
from cards.io.io_manager import IOManager


@pytest.fixture
def batch_size():
    return 100


@pytest.mark.serial
def test_save_serial(
    ctx: ExecutionContext,
    input_shape: tuple[int, ...],
    seed: int,
    tmp_path: Path,
) -> None:
    """
    Test the saving of data for serial settings on CPU.
    """
    rng = xp.random.default_rng(seed)
    x = rng.standard_normal(input_shape)
    y = rng.standard_normal((2, *input_shape))
    z = rng.standard_normal(input_shape)

    io_manager = IOManager(ctx)

    data = {"x": x, "y": y}
    tmp_path_str = tmp_path.as_posix() if tmp_path else ""
    filename = join(tmp_path_str, "dummy_save_data.h5")

    with io_manager.open(filename, "w") as file:
        io_manager.write_dict(file, data)
        io_manager.write_array(file, "z", z)

    with io_manager.open(filename, "r") as file:
        check_x = xp.allclose(file["x"][:], x)
        check_y = xp.allclose(file["y"][:], y)
        check_z = xp.allclose(file["z"][:], z)

    assert check_x and check_y and check_z


@pytest.mark.serial
def test_load_serial(
    ctx: ExecutionContext,
    input_shape: tuple[int, ...],
    seed: int,
    tmp_path: Path,
) -> None:
    """
    Test the loading of data for serial settings on CPU.
    """
    rng = xp.random.default_rng(seed)
    x = rng.standard_normal(input_shape)
    y = rng.standard_normal(np.asarray([2, *input_shape]))
    z = rng.standard_normal(input_shape)

    tmp_path_str = tmp_path.as_posix() if tmp_path else ""
    filename = join(tmp_path_str, "dummy_load_data.h5")

    with h5py.File(filename, "w") as file:
        file["x"] = x.get() if ctx.is_gpu else x
        file["y"] = y.get() if ctx.is_gpu else y
        file["z"] = z.get() if ctx.is_gpu else z

    io_manager = IOManager(ctx)

    with io_manager.open(filename, "r") as file:
        data = io_manager.read_dict(file)

    print(type(data["x"]), type(x))

    checkX = xp.allclose(data["x"], x)
    checkY = xp.allclose(data["y"], y)
    checkZ = xp.allclose(data["z"], z)

    assert checkX and checkY and checkZ


@pytest.mark.mpi
def test_mpi_save_and_load(
    ctx: ExecutionContext,
    input_shape: tuple[int, ...],
    seed: int,
    tmp_path: Path,
) -> None:
    """
    Test the distributed saving and loading of data for CPU settings.
    """
    sync_comm = SyncCartesianCommunicator(
        ctx.comm,
        np.asarray(ctx.generate_grid_shape(3)),
        np.asarray(input_shape),
        np.zeros(len(input_shape)),
        np.zeros(len(input_shape)),
    )
    s = sync_comm.cartslicer.slice_global_buffer_to_tile
    tile_size = sync_comm.cartslicer.tile_size

    rng = xp.random.default_rng([ctx.rank, seed])
    tile_x = rng.standard_normal(size=tile_size)

    io_manager = IOManager(ctx)

    if ctx.is_master:
        filename = str(tmp_path / "distributed_save_data.h5")
    else:
        filename = None
    filename = ctx.comm.bcast(filename, root=0)

    data = {"y": tile_x}
    slices = {"x": s, "y": s}
    global_shapes = {"y": input_shape}

    with io_manager.open(filename, "w") as file:
        io_manager.write_array(file, "x", tile_x, input_shape, s)
        io_manager.write_dict(
            file,
            data_dict=data,
            global_shapes=global_shapes,
            slices=slices,
        )

    with io_manager.open(filename, "r") as file:
        data_read = io_manager.read_dict(file, slices=slices)

    print(data_read["x"].shape, tile_x.shape)

    check_x = xp.array_equal(tile_x, data_read["x"])
    check_y = xp.array_equal(tile_x, data_read["y"])

    assert check_x and check_y


@pytest.mark.cpu
def test_save_and_load_cpu_rng(
    ctx: ExecutionContext,
    input_shape: tuple[int, ...],
    seed: int,
    tmp_path: Path,
) -> None:
    """
    Test the saving and loading of the state of a random number generator on CPU.
    """
    rng = np.random.default_rng([ctx.rank, seed])
    n_trials = 2

    for _ in range(n_trials):
        rng.standard_normal(input_shape)

    io_manager = IOManager(ctx)

    if ctx.is_mpi:
        if ctx.is_master:
            filename = str(tmp_path / "save_rng_cpu.h5")
        else:
            filename = None
        filename = ctx.comm.bcast(filename, root=0)
    else:
        filename = str(tmp_path / "save_rng_cpu.h5")

    with io_manager.open(filename, "w") as file:
        io_manager.write_rng(file, rng)

    with io_manager.open(filename, "r") as file:
        rng2 = io_manager.read_rng(file)

    check = np.zeros(n_trials, dtype=bool)

    for i in range(n_trials):
        check[i] = np.array_equal(
            rng.standard_normal(input_shape),
            rng2.standard_normal(input_shape),  # type:ignore
        )

    assert check.all()


@pytest.mark.gpu
def test_save_and_load_gpu_rng(
    ctx: ExecutionContext,
    input_shape: tuple[int, ...],
    seed: int,
    tmp_path: Path,
) -> None:
    """
    Test the saving and loading of the state of a random number generator on GPU.
    """
    import torch

    rng = torch.Generator("cuda").manual_seed(ctx.rank * 42)
    n_trials = 2

    for _ in range(n_trials):
        torch.rand(input_shape, generator=rng, device="cuda")

    io_manager = IOManager(ctx)

    if ctx.is_mpi:
        if ctx.is_master:
            filename = str(tmp_path / "save_rng_gpu.h5")
        else:
            filename = None
        filename = ctx.comm.bcast(filename, root=0)
    else:
        filename = str(tmp_path / "save_rng_gpu.h5")

    with io_manager.open(filename, "w") as file:
        io_manager.write_rng(file, rng)

    with io_manager.open(filename, "r") as file:
        rng2 = io_manager.read_rng(file)

    check = np.zeros(n_trials, dtype=bool)

    for i in range(n_trials):
        check[i] = torch.equal(
            torch.rand(input_shape, generator=rng, device="cuda"),
            torch.rand(input_shape, generator=rng2, device="cuda"),  # type:ignore
        )

    assert check.all()
