r"""Define useful function to run a sample on the inpainting problem."""

from os.path import join
import h5py

import numpy as np
import cupy as cp

from mpi4py import MPI

import logging

from mcmc.models.DistributedGaussianInpainting import DistributedGaussianInpaintingModel
from mcmc.models.GaussianInpaintingModel import GaussianInpaintingModel
from mcmc.models.GpuGaussianInpaintingModel import GpuGaussianInpaintingModel
from mcmc.sampler.DistributedSampler import DistributedSampler
from mcmc.sampler.GpuSampler import GpuSampler
from mcmc.sampler.SerialSampler import Sampler
from mcmc.slicer.cartesian_comm_slicer import CartesianCommSlicer
from mcmc.TransitionKernel.GpuTransitionKernel import GpuPSGLA
from mcmc.TransitionKernel.TransitionKernel import PSGLA


def load_from_h5(filename):
    """load the mask, sig2 and data entries from the h5 file"""
    with h5py.File(filename, "r") as data_file:
        mask = data_file["mask"][:]
        sigma2 = data_file[
            "sig2"
        ][
            ()
        ]  # ! name temporarily modified for compatibility with data generated with older library
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

    X = GpuPSGLA(observations.shape, cp.asarray(step_size_X))
    Z = GpuPSGLA((2, *X.current_state.shape), cp.asarray(step_size_Z))

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
        img_size, grid_size, observations, mask, X, Z, sigma2, reg_coef, split_coef
    )

    sampler = DistributedSampler(
        comm, batch_size, nb_batches, seed, "sample", save_path, model, logger
    )

    sampler.sample()


def resume_serial_sampler(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    batch_size: int,
    nb_batches: int,
    seed: int,
    save_path: str,
    data_path: str,
    restart_batch: int,
    resume_save_path: str,
):
    r"""resume_serial_sampler Resume the generation of a sample, with the serial implementation.

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
    restart_batch : int
        Number of the batch from which we will resume sampling.
    resume_save_path : str
        Path to the folder where we will save the data generated form the resumed run.
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

    resume_file_name = join(save_path, "sample" + str(restart_batch) + ".h5")
    sampler.restart(resume_file_name, restart_batch, resume_save_path)
    sampler.sample()


def resume_gpu_sampler(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    batch_size: int,
    nb_batches: int,
    seed: int,
    save_path: str,
    data_path: str,
    restart_batch: int,
    resume_save_path: str,
):
    r"""resume_gpu_sampler  Resume the generation of a sample, with the implementation on GPU.

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
    restart_batch : int
        Number of the batch from which we will resume sampling.
    resume_save_path : str
        Path to the folder where we will save the data generated form the resumed run.
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

    resume_file_name = join(save_path, "sample" + str(restart_batch) + ".h5")
    sampler.restart(resume_file_name, restart_batch, resume_save_path)
    sampler.sample()


def resume_distributed_sampler(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    batch_size: int,
    nb_batches: int,
    seed: int,
    save_path: str,
    data_path: str,
    restart_batch: int,
    resume_save_path: str,
):
    r"""resume_distributed_sampler Resume the generation of a sample, with the distributed implementation.

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
    restart_batch : int
        Number of the batch from which we will resume sampling.
    resume_save_path : str
        Path to the folder where we will save the data generated form the resumed run.
    """

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
        img_size, grid_size, observations, mask, X, Z, sigma2, reg_coef, split_coef
    )

    sampler = DistributedSampler(
        comm, batch_size, nb_batches, seed, "sample", save_path, model, logger
    )
    sampler.sample()

    resume_file_name = join(save_path, "sample" + str(restart_batch) + ".h5")
    sampler.restart(resume_file_name, restart_batch, resume_save_path)
    sampler.sample()
