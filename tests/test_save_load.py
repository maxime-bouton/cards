r"""Test the writting/loading of data on/from disk memory."""

import numpy as np
from mcmc.DataManager.DataManager import DataManager
import h5py
from os.path import join
import sys

import pytest


@pytest.fixture
def dims():
    return np.asarray([100, 50], dtype=int)


@pytest.fixture
def seed():
    return 1234


@pytest.fixture
def new_seed():
    return 4321


def test_save(tmp_path, dims):
    rng = np.random.default_rng(1234)
    X = rng.standard_normal(dims)
    Y = rng.standard_normal(np.asarray([2, *dims]))
    Z = rng.standard_normal(dims)

    data_manager = DataManager()

    data = {}
    data["x"] = X
    data["y"] = Y

    if tmp_path is not None:
        tmp_path_str = tmp_path.as_posix()
    else:
        tmp_path_str = ""
    filename = join(tmp_path_str, "dummy_save_data.h5")

    with h5py.File(filename, "w") as file:
        data_manager.save_dict(data, file)
        data_manager.save_array(Z, file, "z")

    with h5py.File(filename, "r") as file:
        checkX = (file["x"][:] == X).all()
        checkY = (file["y"][:] == Y).all()
        checkZ = (file["z"][:] == Z).all()

    assert checkX and checkY and checkZ


def test_load(tmp_path, dims):
    rng = np.random.default_rng(1234)
    X = rng.standard_normal(dims)
    Y = rng.standard_normal(np.asarray([2, *dims]))
    Z = rng.standard_normal(dims)

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

    data = data_manager.load_h5(filename)

    checkX = (data["x"] == X).all()
    checkY = (data["y"] == Y).all()
    checkZ = (data["z"] == Z).all()

    assert checkX and checkY and checkZ


def test_write_read_rng(tmp_path, dims, seed, new_seed):
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(new_seed)

    for i in range(10):
        rng.standard_normal(dims)

    data_manager = DataManager()
    if tmp_path is not None:
        tmp_path_str = tmp_path.as_posix()
    else:
        tmp_path_str = ""
    filename = join(tmp_path_str, "test_rng.h5")

    with h5py.File(filename, "w") as file:
        data_manager.save_rng(rng, file)

    data = data_manager.load_h5(filename)

    state_array = data["rng_state_array"]
    inc_array = data["rng_inc_array"]

    new_state = int.from_bytes(state_array, sys.byteorder)
    new_inc = int.from_bytes(inc_array, sys.byteorder)

    new_rng_state = rng.bit_generator.__getstate__()
    new_rng_state[0]["state"]["state"] = new_state
    new_rng_state[0]["state"]["inc"] = new_inc

    rng2.bit_generator.__setstate__(new_rng_state)

    check = np.zeros(10, dtype=bool)

    for i in range(10):
        check[i] = (rng.standard_normal(dims) == rng2.standard_normal(dims)).all()

    assert check.all()


if __name__ == "__main__":
    tmp_path = None
    seed = 1234
    new_seed = 4321

    test_save(tmp_path, dims)
    test_load(tmp_path, dims)
    test_write_read_rng(tmp_path, dims, seed, new_seed)
