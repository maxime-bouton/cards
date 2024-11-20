r"""Test the writting/loading of data on/from disk memory."""

from os.path import join

import h5py
import numpy as np
import pytest

from mcmc.DataManager.DataManager import DataManager

pytestmark = pytest.mark.numpy


@pytest.fixture
def dims():
    return np.asarray([100, 50], dtype=int)


@pytest.fixture
def seed():
    return 1234


@pytest.fixture
def new_seed():
    return 4321


# FIXME: missing docstrings
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
        checkX = np.allclose(file["x"][:], X)
        checkY = np.allclose(file["y"][:], Y)
        checkZ = np.allclose(file["z"][:], Z)

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
    with h5py.File(filename, "r") as file:
        data = data_manager.load_h5(file)

    checkX = np.allclose(data["x"], X)
    checkY = np.allclose(data["y"], Y)
    checkZ = np.allclose(data["z"], Z)

    assert checkX and checkY and checkZ


def test_write_read_rng(tmp_path, dims, seed, new_seed):
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(new_seed)
    n_trials = 10

    for i in range(n_trials):
        rng.standard_normal(dims)

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
        check[i] = np.allclose(rng.standard_normal(dims), rng2.standard_normal(dims))

    assert check.all()


if __name__ == "__main__":
    tmp_path = None
    dims = np.array([100, 50], dtype=int)
    seed = 1234
    new_seed = 4321

    test_save(tmp_path, dims)
    test_load(tmp_path, dims)
    test_write_read_rng(tmp_path, dims, seed, new_seed)
