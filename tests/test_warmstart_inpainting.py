r"""Testing sampler warmstart functionality on a Gaussian inpainting model."""

from os import path

import h5py
import numpy as np

from mcmc.models.GaussianInpaintingModel import GaussianInpaintingModel
from mcmc.sampler.SerialSampler import Sampler
from mcmc.TransitionKernel.TransitionKernel import PSGLA


def test_warmstart_inpainting(tmp_path):
    nb_batches = 10
    batch_size = 100
    save_path = tmp_path / "sample/"
    save_path.mkdir()
    restart_save_path = tmp_path / "resumed/"
    restart_save_path.mkdir()
    num_batch_load = 4

    split_coeff = 1
    reg_coeff = 1
    seed = 2345

    sigma2 = 1.5

    M, N = 20, 20

    rng = np.random.default_rng(1234)
    mask = rng.binomial(1, 0.4, [M, N])
    observations = np.ones([M, N]) * mask + rng.normal(0, np.sqrt(sigma2), [M, N])

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
    X = np.zeros([nb_batches, M, N])
    resumed_potential = []
    resumed_X = np.zeros([nb_batches, M, N])

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
