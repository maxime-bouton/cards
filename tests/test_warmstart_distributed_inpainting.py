r"""Testing sampler warmstart functionality on a Gaussian inpainting model with the distributed implementation."""

import numpy as np
import pytest

from mcmc.models.DistributedGaussianInpainting import DistributedGaussianInpaintingModel
from mcmc.sampler.DistributedSampler import DistributedSampler
from mcmc.TransitionKernel.TransitionKernel import PSGLA
from mcmc.slicer.cartesian_comm_slicer import CartesianCommSlicer
from mpi4py import MPI

import os
import shutil


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


def test_warmstart_distributed_inpainting(nb_batches, batch_size, dims, seed):
    tmp_path = "./"
    rank = MPI.COMM_WORLD.Get_rank()
    grid_size = np.asarray(MPI.Compute_dims(MPI.COMM_WORLD.Get_size(), 2), dtype=int)
    mpi_cart_comm = MPI.COMM_WORLD.Create_cart(grid_size)
    ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))

    save_path = os.path.join(tmp_path, "sample/")
    restart_save_path = os.path.join(tmp_path, "resumed/")
    # num_batch_load = 4

    if rank == 0:
        os.mkdir(save_path)
        os.mkdir(restart_save_path)

        for i in range(nb_batches):
            file = open(os.path.join(save_path, "sample" + str(i) + "h5"), "w")
            file.close()

    split_coeff = 1
    reg_coeff = 1

    sigma2 = 1.5

    slicer = CartesianCommSlicer(
        ranknd, grid_size, dims, np.asarray([0, 0]), np.asarray([0, 0])
    )

    local_dims = slicer.tile_size

    rng = np.random.default_rng(seed)
    mask = rng.binomial(1, 0.4, local_dims)
    observations = np.ones(local_dims) * mask + rng.normal(
        0, np.sqrt(sigma2), local_dims
    )

    step_size_X = 0.99 * 1.0 / (8.0 / split_coeff + 1.0 / sigma2)
    X = PSGLA(local_dims, step_size_X)

    step_size_Z = 0.99 / split_coeff
    Z = PSGLA((2, *local_dims), step_size_Z)

    model = DistributedGaussianInpaintingModel(
        dims, grid_size, observations, mask, X, Z, sigma2, reg_coeff, split_coeff
    )

    sampler = DistributedSampler(
        batch_size, nb_batches, seed, "sample", str(save_path), model
    )

    # first run
    sampler.sample()

    # load_path = path.join(str(save_path), "sample{}.h5".format(num_batch_load - 1))
    # sampler.restart(load_path, str(restart_save_path), num_batch_load)

    # resumed run
    # sampler.sample()

    shutil.rmtree(save_path)
    shutil.rmtree(restart_save_path)
