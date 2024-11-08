r"""Testing sampler warmstart functionality on a Gaussian inpainting model."""

from os import path

import h5py
import numpy as np
import pytest

from mcmc.models.GaussianInpaintingModel import GaussianInpaintingModel
from mcmc.sampler.SerialSampler import Sampler
from mcmc.TransitionKernel.TransitionKernel import PSGLA


@pytest.fixture
def nb_batches():
    return 10


@pytest.fixture
def batch_size():
    return 100


@pytest.fixture
def dims():
    return np.asarray([20, 20], dtype=int)


@pytest.fixture
def seed():
    return 1234


def test_warmstart_inpainting(tmp_path, nb_batches, batch_size, dims, seed):
    if tmp_path is not None:
        save_path = tmp_path / "sample/"
        save_path.mkdir()
        restart_save_path = tmp_path / "resumed/"
        restart_save_path.mkdir()
    else:
        from pathlib import Path

        save_path = "sample/"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        restart_save_path = "resumed/"
        Path(restart_save_path).mkdir(parents=True, exist_ok=True)

    num_batch_load = 4

    split_coeff = 1
    reg_coeff = 1

    sigma2 = 1.5

    rng = np.random.default_rng(1234)
    mask = rng.binomial(1, 0.4, dims)
    observations = np.ones(dims) * mask + rng.normal(0, np.sqrt(sigma2), dims)

    step_size_X = 0.99 * 1.0 / (8.0 / split_coeff + 1.0 / sigma2)
    X = PSGLA(observations.shape, step_size_X)

    step_size_Z = 0.99 / split_coeff
    Z = PSGLA((2, *X.current_state.shape), step_size_Z)

    model = GaussianInpaintingModel(
        observations, mask, X, Z, sigma2, reg_coeff, split_coeff
    )

    sampler = Sampler(batch_size, nb_batches, seed, "sample", str(save_path), model)

    sampler.sample()

    load_path = path.join(str(save_path), "sample{}.h5".format(num_batch_load - 1))
    sampler.restart(load_path, num_batch_load, str(restart_save_path))

    sampler.sample()

    potential = []
    X = np.zeros([nb_batches, *dims])
    resumed_potential = []
    resumed_X = np.zeros([nb_batches, *dims])

    for i in range(num_batch_load, 10):
        with h5py.File(str(save_path) + "/sample" + str(i) + ".h5") as file:
            potential = np.append(potential, np.asarray(file["potential"][:]))
            X[i, ...] = np.asarray(file["X"][:])

        with h5py.File(str(restart_save_path) + "/sample" + str(i) + ".h5") as file:
            resumed_potential = np.append(
                resumed_potential, np.asarray(file["potential"][:])
            )
            resumed_X[i] = np.asarray(file["X"][:])

    assert np.allclose(potential, resumed_potential) and np.allclose(X, resumed_X)


if __name__ == "__main__":
    nb_batches = 10
    batch_size = 100
    dims = np.asarray([20, 20], dtype=int)
    seed = 1234
    tmp_path = None

    test_warmstart_inpainting(tmp_path, nb_batches, batch_size, dims, seed)
