r"""Testing sampler warmstart functionality on a Gaussian inpainting model with the distributed implementation."""

import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
from mpi4py import MPI

from mcmc.models.DistributedGaussianInpainting import DistributedGaussianInpaintingModel
from mcmc.sampler.DistributedSampler import DistributedSampler
from mcmc.slicer.cartesian_comm_slicer import CartesianCommSlicer
from mcmc.TransitionKernel.TransitionKernel import PSGLA

pytestmark = [pytest.mark.mpi, pytest.mark.numpy]


@pytest.fixture
def nb_batches():
    return 5


@pytest.fixture
def num_batch_load():
    return 3


@pytest.fixture
def batch_size():
    return 1000


@pytest.fixture
def dims():
    return np.asarray([20, 20], dtype=int)


@pytest.fixture
def seed():
    return 1234


# FIXME: missing docstrings
def test_warmstart_distributed_inpainting(
    nb_batches, num_batch_load, batch_size, dims, seed
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    grid_size = np.asarray(MPI.Compute_dims(comm.Get_size(), 2), dtype=int)
    mpi_cart_comm = comm.Create_cart(grid_size)
    ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))

    tmp_path = "./"
    save_path = os.path.join(tmp_path, "sample/")
    restart_save_path = os.path.join(tmp_path, "resumed/")

    if rank == 0:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        Path(restart_save_path).mkdir(parents=True, exist_ok=True)

    split_coeff = 1
    reg_coeff = 1
    sigma2 = 1.5

    slicer = CartesianCommSlicer(
        ranknd, grid_size, dims, np.asarray([0, 0]), np.asarray([0, 0])
    )
    local_dims = slicer.tile_size

    # generate noisy data on each worker
    if rank == 0:
        ss = np.random.SeedSequence(seed)
        # spawn off nworkers child SeedSequences to pass to child processes.
        child_seed = np.array(ss.spawn(comm.Get_size()))
    else:
        child_seed = None
    local_seed = comm.scatter(child_seed, root=0)
    rng = np.random.default_rng(local_seed)

    mask = rng.binomial(1, 0.4, local_dims)
    observations = np.ones(local_dims) * mask + rng.normal(
        0, np.sqrt(sigma2), local_dims
    )

    # instantiate Gaussian inpainting model and sampler
    step_size_X = 0.99 * 1.0 / (8.0 / split_coeff + 1.0 / sigma2)
    X = PSGLA(local_dims, step_size_X)

    step_size_Z = 0.99 / split_coeff
    Z = PSGLA((2, *local_dims), step_size_Z)

    model = DistributedGaussianInpaintingModel(
        dims, grid_size, observations, mask, X, Z, sigma2, reg_coeff, split_coeff
    )

    sampler = DistributedSampler(
        comm, batch_size, nb_batches, seed, "sample", str(save_path), model
    )

    # first run
    sampler.sample()

    # resumed run
    load_filename = os.path.join(save_path, "sample" + str(num_batch_load)) + ".h5"
    sampler.restart(load_filename, restart_save_path, num_batch_load)
    sampler.sample()

    # check consistency of samples and values for the potential function
    # FIXME: test should be done in parallel, not only on worker 0
    if rank == 0:
        X = np.zeros((nb_batches - num_batch_load, *dims))
        resumed_X = np.zeros((nb_batches - num_batch_load, *dims))
        potential = np.zeros(((nb_batches - num_batch_load) * batch_size,))
        resumed_potential = np.zeros(((nb_batches - num_batch_load) * batch_size,))

        for i in range(num_batch_load + 1, nb_batches + 1):
            j = i - (num_batch_load + 1)

            with h5py.File(save_path + "/sample" + str(i) + ".h5") as file:
                potential[j * batch_size : (j + 1) * batch_size] = file["potential"][:]
                X[j] = file["X"][:]

            with h5py.File(restart_save_path + "/sample" + str(i) + ".h5") as file:
                resumed_potential[j * batch_size : (j + 1) * batch_size] = file[
                    "potential"
                ][:]
                resumed_X[j] = file["X"][:]

        test_check = np.allclose(potential, resumed_potential) and np.allclose(
            X, resumed_X
        )

    if rank == 0:
        assert test_check
        shutil.rmtree(save_path)
        shutil.rmtree(restart_save_path)


if __name__ == "__main__":
    nb_batches = 5
    num_batch_load = 4  # in [1, n_batches]
    batch_size = 100
    dims = np.asarray([20, 20], dtype=int)
    seed = 1234
    test_warmstart_distributed_inpainting(
        nb_batches, num_batch_load, batch_size, dims, seed
    )
