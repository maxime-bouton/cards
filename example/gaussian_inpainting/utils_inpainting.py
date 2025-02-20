import h5py
import cupy as cp
import logging
import json

import numpy as np

from mcmc.models.DistributedGaussianInpainting import DistributedGaussianInpaintingModel
from mcmc.models.GaussianInpaintingModel import GaussianInpaintingModel
from mcmc.models.GpuGaussianInpaintingModel import GpuGaussianInpaintingModel
from mcmc.models.MultiGpuGaussianInpaintingModel import MultiGpuGaussianInpaintingModel
from mcmc.sampler.DistributedSampler import DistributedSampler
from mcmc.sampler.MultiGpuSampler import MultiGpuSampler
from mcmc.sampler.GpuSampler import GpuSampler
from mcmc.sampler.SerialSampler import Sampler
from mcmc.slicer.cartesian_comm_slicer import CartesianCommSlicer
from mcmc.TransitionKernel.GpuTransitionKernel import GpuPSGLA
from mcmc.TransitionKernel.GpuTransitionKernel import MultiGpuPSGLA
from mcmc.TransitionKernel.TransitionKernel import PSGLA
from mcmc.operators.inpainting_v2 import SerialInpainting
from mcmc.utils.utils import load_img_size, generate_observations, apply_gaussian_noise


def load_from_h5(filename):
    """load the mask01, sig2 and data entries from the h5 file"""
    with h5py.File(filename, "r") as data_file:
        mask = data_file["mask01"][:]
        sigma2 = data_file["sigma2"][()]
        observations = data_file["data"][:]
    return mask, sigma2, observations


def compute_step_size(split_coef: float, sigma2: float):
    x = 0.99 * 1.0 / (8.0 / split_coef + 1.0 / sigma2)
    z = 0.99 * split_coef
    return (x, z)


def compute_serial(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    batch_size: int,
    nb_batches: int,
    seed: int,
    save_path: str,
    data_path: str,
):
    r"""compute_serial Generate a sample with the serial implementation.

    Parameters
    ----------
    logger: logging.Logger
        Logger object.
    split_coef : float
        Splitting coefficient, parameter of the model.
    reg_coef : float
        Regularization coefficient, parameter of the model.
    batch_size : float
        Number of iterations for a batch.
    nb_batches : int
        Number of batches of the sample.
    seed : int
        Seed.
    save_path : str
        Path to the directory where we will save the sample.
    data_path : str
        Path to the file containing the deteriorated signal.
    """

    mask, sigma2, observations = load_from_h5(data_path)

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2)
    X = PSGLA(observations.shape, step_size_X)
    Z = PSGLA((2, *X.current_state.shape), step_size_Z)

    model = GaussianInpaintingModel(
        observations, mask, X, Z, sigma2, reg_coef, split_coef
    )

    sampler = Sampler(batch_size, nb_batches, seed, "sample", save_path, model, logger)

    sampler.sample()


def compute_gpu(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    batch_size: int,
    nb_batches: int,
    seed: int,
    save_path: str,
    data_path: str,
):
    r"""compute_gpu  Generate a sample with the implementation on GPU.

    Parameters
    ----------
    logger: logging.Logger
        Logger object.
    split_coef : float
        Splitting coefficient, parameter of the model.
    reg_coef : float
        Regularization coefficient, parameter of the model.
    batch_size : float
        Number of iterations for a batch.
    nb_batches : int
        Number of batches of the sample.
    seed : int
        Seed.
    save_path : str
        Path to the directory where we will save the sample.
    data_path : str
        Path to the file containing the deteriorated signal.
    """

    mask, sigma2, observations = load_from_h5(data_path)

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2)

    X = GpuPSGLA(observations.shape, step_size_X)
    Z = GpuPSGLA((2, *X.current_state.shape), step_size_Z)

    model = GpuGaussianInpaintingModel(
        cp.asarray(observations),
        cp.asarray(mask),
        X,
        Z,
        sigma2,
        reg_coef,
        split_coef,
    )

    sampler = GpuSampler(
        batch_size, nb_batches, seed, "sample", save_path, model, logger
    )

    sampler.sample()


def compute_distributed(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    batch_size: int,
    nb_batches: int,
    seed: int,
    save_path: str,
    data_path: str,
):
    r"""compute_distributed Generate a sample with the distributed implementation.

    Parameters
    ----------
    logger: logging.Logger
        Logger object.
    split_coef : float
        Splitting coefficient, parameter of the model.
    reg_coef : float
        Regularization coefficient, parameter of the model.
    batch_size : int
        Number of iterations for a batch.
    nb_batches : int
        Number of batches of the sample.
    seed : int
        Seed.
    save_path : str
        Path to the directory where we will save the sample.
    data_path : str
        Path to the file containing the deteriorated signal.
    """
    import numpy as np
    from mpi4py import MPI

    mask, sigma2, observations = load_from_h5(data_path)
    img_size = observations.shape

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    grid_size = np.asarray(MPI.Compute_dims(comm.Get_size(), 2), dtype=int)
    mpi_cart_comm = comm.Create_cart(grid_size)
    ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))

    slicer = CartesianCommSlicer(
        ranknd, grid_size, observations.shape, np.asarray([0, 0]), np.asarray([0, 0])
    )
    tile_size = slicer.tile_size
    mask = mask[slicer.slice_global_buffer_to_tile]
    observations = observations[slicer.slice_global_buffer_to_tile]

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2)
    X = PSGLA(tile_size, step_size_X)
    Z = PSGLA((2, *tile_size), step_size_Z)

    model = DistributedGaussianInpaintingModel(
        comm,
        img_size,
        grid_size,
        observations,
        mask,
        X,
        Z,
        sigma2,
        reg_coef,
        split_coef,
    )

    sampler = DistributedSampler(
        comm, batch_size, nb_batches, seed, "sample", save_path, model, logger
    )

    sampler.sample()


def compute_multi_gpu(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    batch_size: int,
    nb_batches: int,
    seed: int,
    save_path: str,
    data_path: str,
):
    r"""compute_distributed Generate a sample with the distributed implementation.

    Parameters
    ----------
    logger: logging.Logger
        Logger object.
    split_coef : float
        Splitting coefficient, parameter of the model.
    reg_coef : float
        Regularization coefficient, parameter of the model.
    batch_size : int
        Number of iterations for a batch.
    nb_batches : int
        Number of batches of the sample.
    seed : int
        Seed.
    save_path : str
        Path to the directory where we will save the sample.
    data_path : str
        Path to the file containing the deteriorated signal.
    """
    import numpy as np
    from mpi4py import MPI

    mask01, sigma2, obs = load_from_h5(data_path)
    img_size = obs.shape

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    grid_size = np.asarray(MPI.Compute_dims(comm.Get_size(), 2), dtype=int)
    mpi_cart_comm = comm.Create_cart(grid_size)
    ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))

    slicer = CartesianCommSlicer(
        ranknd, grid_size, obs.shape, np.asarray([0, 0]), np.asarray([0, 0])
    )
    tile_size = slicer.tile_size
    with cp.cuda.Device(rank):
        mask = cp.asarray(mask01[slicer.slice_global_buffer_to_tile])
        observations = cp.asarray(obs[slicer.slice_global_buffer_to_tile])

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2)
    X = MultiGpuPSGLA(tile_size, step_size_X, rank)
    Z = MultiGpuPSGLA((2, *tile_size), step_size_Z, rank)

    model = MultiGpuGaussianInpaintingModel(
        comm,
        img_size,
        grid_size,
        observations,
        mask,
        X,
        Z,
        sigma2,
        reg_coef,
        split_coef,
    )

    sampler = MultiGpuSampler(
        comm, batch_size, nb_batches, seed, "sample", save_path, model, logger
    )

    sampler.sample()


def add_inpainting_params(args: dict, config_file_path: str) -> None:
    config_file = open(config_file_path)
    params = json.load(config_file)

    args["split_coef"] = params["alpha"]
    args["reg_coef"] = params["regularizationCoefficient"]
    return


def generate_inpainting_observations(
    original_path: str,
    mask_loss: float,
    snr: float,
    data_seed: int,
    obs_path: str,
    maximum: float = 1.0,
) -> None:
    dims = load_img_size(original_path)
    rng = np.random.default_rng(data_seed)
    mask = rng.binomial(1, 1 - mask_loss, dims)

    inpainting_operator = SerialInpainting(mask)

    inpainting_params = {}
    inpainting_params["mask"] = mask
    inpainting_params["mask01"] = mask

    generate_observations(
        original_path,
        inpainting_operator,
        snr,
        apply_gaussian_noise,
        data_seed,
        obs_path,
        problem_parameters=inpainting_params,
        maximum=maximum,
    )
