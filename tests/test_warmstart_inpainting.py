r"""Testing sampler warmstart functionality on a Gaussian inpainting model."""

import logging
from os import path

import h5py
import numpy as np
import pytest

from mcmc.models.GaussianInpaintingModel import GaussianInpaintingModel
from mcmc.sampler.SerialSampler import Sampler
from mcmc.TransitionKernel.TransitionKernel import PSGLA

pytestmark = pytest.mark.numpy


@pytest.fixture
def nb_batches():
    return 5


@pytest.fixture
def num_batch_load():
    return 3


@pytest.fixture
def batch_size():
    return 100


@pytest.fixture
def dims():
    return np.asarray([20, 20], dtype=int)


@pytest.fixture
def seed():
    return 1234


def test_warmstart_inpainting(
    tmp_path, nb_batches, num_batch_load, batch_size, dims, seed
):
    if tmp_path is not None:
        save_path = tmp_path / "sample/"
        save_path.mkdir()
        restart_save_path = tmp_path / "resumed/"
        restart_save_path.mkdir()
        logfilename = path.join(tmp_path.as_posix(), "warmstart_inpainting.log")

    else:
        from pathlib import Path

        save_path = "sample"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        restart_save_path = "resumed"
        Path(restart_save_path).mkdir(parents=True, exist_ok=True)
        logfilename = "warmstart_inpainting.log"

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=logfilename,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
    )

    split_coeff = 1
    reg_coeff = 1

    sigma2 = 1.5

    # generate noisy data
    rng = np.random.default_rng(1234)
    mask = rng.binomial(1, 0.4, dims)
    observations = np.ones(dims) * mask + rng.normal(0, np.sqrt(sigma2), dims)

    # instantiate Gaussian inpainting model and sampler
    step_size_X = 0.99 * 1.0 / (8.0 / split_coeff + 1.0 / sigma2)
    X = PSGLA(observations.shape, step_size_X)

    step_size_Z = 0.99 / split_coeff
    Z = PSGLA((2, *X.current_state.shape), step_size_Z)

    model = GaussianInpaintingModel(
        observations, mask, X, Z, sigma2, reg_coeff, split_coeff
    )

    sampler = Sampler(
        batch_size, nb_batches, seed, "sample", str(save_path), model, logger
    )

    # first run
    sampler.sample()

    # resumed run
    load_filename = path.join(str(save_path), "sample" + str(num_batch_load)) + ".h5"
    sampler.restart(load_filename, num_batch_load, str(restart_save_path))
    sampler.sample()

    # check consistency of samples and values for the potential function
    X = np.zeros((nb_batches - num_batch_load, *dims))
    resumed_X = np.zeros((nb_batches - num_batch_load, *dims))
    potential = np.zeros(((nb_batches - num_batch_load) * batch_size,))
    resumed_potential = np.zeros(((nb_batches - num_batch_load) * batch_size,))

    for i in range(num_batch_load + 1, nb_batches + 1):
        j = i - (num_batch_load + 1)

        with h5py.File(str(save_path) + "/sample" + str(i) + ".h5") as file:
            potential[j * batch_size : (j + 1) * batch_size] = file["potential"][:]
            X[j] = file["X"][:]

        with h5py.File(str(restart_save_path) + "/sample" + str(i) + ".h5") as file:
            resumed_potential[j * batch_size : (j + 1) * batch_size] = file[
                "potential"
            ][:]
            resumed_X[j] = file["X"][:]

    assert np.allclose(potential, resumed_potential) and np.allclose(X, resumed_X)


if __name__ == "__main__":
    nb_batches = 5
    num_batch_load = 4  # in [1, n_batches]
    batch_size = 100
    dims = np.asarray([20, 20], dtype=int)
    seed = 1234
    tmp_path = None

    test_warmstart_inpainting(
        tmp_path, nb_batches, num_batch_load, batch_size, dims, seed
    )
