import logging
from os.path import join

import cupy as cp
import h5py

from mcmc.models.base_gaussian_inpainting_model import (
    DistributedGaussianInpaintingTvModel,
    GaussianInpaintingTvModel,
    GaussianInpaintingTvParameters,
)
from mcmc.sampler.base_sampler import SamplerParameters
from mcmc.sampler.distributed_sampler import DistributedSampler
from mcmc.sampler.gpu_sampler import GpuSampler
from mcmc.sampler.multi_gpu_sampler import MultiGpuSampler
from mcmc.sampler.serial_sampler import SerialSampler
from mcmc.slicer.cartesian_comm_slicer import CartesianCommSlicer
from mcmc.transition_kernel.gpu_psgla import GpuPSGLA
from mcmc.transition_kernel.psgla import PSGLA


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


#! the backend must be set before importing/calling any of the compute funtions
def compute_serial(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    sampler_params: SamplerParameters,
    data_path: str,
):
    mask, sigma2, observations = load_from_h5(data_path)

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2)
    X = PSGLA(observations.shape, step_size_X)
    Z = PSGLA((2, *X.current_state.shape), step_size_Z)

    model_params = GaussianInpaintingTvParameters(
        observations, mask, sigma2, reg_coef, split_coef
    )

    model = GaussianInpaintingTvModel(
        model_params,
        X,
        Z,
    )

    sampler = SerialSampler(sampler_params, model, logger)

    sampler.sample()


def resume_serial_sampler(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    sampler_params: SamplerParameters,
    data_path: str,
    restart_batch: int,
    resume_save_path: str,
):
    mask, sigma2, observations = load_from_h5(data_path)

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2)
    X = PSGLA(observations.shape, step_size_X)
    Z = PSGLA((2, *X.current_state.shape), step_size_Z)

    model_params = GaussianInpaintingTvParameters(
        observations, mask, sigma2, reg_coef, split_coef
    )

    model = GaussianInpaintingTvModel(
        model_params,
        X,
        Z,
    )

    sampler = SerialSampler(sampler_params, model, logger)

    resume_file_name = join(
        sampler_params.save_path, "sample" + str(restart_batch) + ".h5"
    )
    sampler.restart(resume_file_name, restart_batch, resume_save_path)
    sampler.sample()


def compute_gpu(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    sampler_params: SamplerParameters,
    data_path: str,
):
    mask, sigma2, observations = load_from_h5(data_path)

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2)

    X = GpuPSGLA(observations.shape, step_size_X)
    Z = GpuPSGLA((2, *X.current_state.shape), step_size_Z)

    model_params = GaussianInpaintingTvParameters(
        cp.asarray(observations), cp.asarray(mask), sigma2, reg_coef, split_coef
    )

    model = GaussianInpaintingTvModel(
        model_params,
        X,
        Z,
    )

    sampler = GpuSampler(sampler_params, model, logger)

    sampler.sample()


def resume_gpu_sampler(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    sampler_params: SamplerParameters,
    data_path: str,
    restart_batch: int,
    resume_save_path: str,
):
    mask, sigma2, observations = load_from_h5(data_path)

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2)

    X = GpuPSGLA(observations.shape, step_size_X)
    Z = GpuPSGLA((2, *X.current_state.shape), step_size_Z)

    model_params = GaussianInpaintingTvParameters(
        cp.asarray(observations), cp.asarray(mask), sigma2, reg_coef, split_coef
    )

    model = GaussianInpaintingTvModel(
        model_params,
        X,
        Z,
    )

    sampler = GpuSampler(sampler_params, model, logger)

    resume_file_name = join(
        sampler_params.save_path, "sample" + str(restart_batch) + ".h5"
    )
    sampler.restart(resume_file_name, restart_batch, resume_save_path)
    sampler.sample()


def compute_distributed(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    sampler_params: SamplerParameters,
    data_path: str,
):
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

    model_params = GaussianInpaintingTvParameters(
        observations, mask, sigma2, reg_coef, split_coef
    )

    model = DistributedGaussianInpaintingTvModel(
        comm,
        img_size,
        grid_size,
        model_params,
        X,
        Z,
    )

    sampler = DistributedSampler(comm, sampler_params, model, logger)

    sampler.sample()


def resume_distributed_sampler(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    sampler_params: SamplerParameters,
    data_path: str,
    restart_batch: int,
    resume_save_path: str,
):
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

    model_params = GaussianInpaintingTvParameters(
        observations, mask, sigma2, reg_coef, split_coef
    )

    model = DistributedGaussianInpaintingTvModel(
        comm,
        img_size,
        grid_size,
        model_params,
        X,
        Z,
    )

    sampler = DistributedSampler(comm, sampler_params, model, logger)

    resume_file_name = join(
        sampler_params.save_path, "sample" + str(restart_batch) + ".h5"
    )
    sampler.restart(resume_file_name, restart_batch, resume_save_path)
    sampler.sample()


def compute_multi_gpu(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    sampler_params: SamplerParameters,
    data_path: str,
):
    import numpy as np
    from mpi4py import MPI

    mask01, sigma2, obs = load_from_h5(data_path)
    img_size = obs.shape

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    grid_size = np.asarray(MPI.Compute_dims(comm.Get_size(), 2), dtype=int)
    mpi_cart_comm = comm.Create_cart(grid_size)
    ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))

    nb_gpu = cp.cuda.runtime.getDeviceCount()
    gpu_id = rank % nb_gpu

    slicer = CartesianCommSlicer(
        ranknd, grid_size, obs.shape, np.asarray([0, 0]), np.asarray([0, 0])
    )
    tile_size = slicer.tile_size

    with cp.cuda.Device(gpu_id):
        mask = cp.asarray(mask01[slicer.slice_global_buffer_to_tile])
        observations = cp.asarray(obs[slicer.slice_global_buffer_to_tile])

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2)
    X = GpuPSGLA(tile_size, step_size_X, gpu_id)
    Z = GpuPSGLA((2, *tile_size), step_size_Z, gpu_id)

    with cp.cuda.Device(gpu_id):
        model_params = GaussianInpaintingTvParameters(
            observations, mask, sigma2, reg_coef, split_coef
        )

    model = DistributedGaussianInpaintingTvModel(
        comm,
        img_size,
        grid_size,
        model_params,
        X,
        Z,
        gpu_id,
    )

    sampler = MultiGpuSampler(comm, sampler_params, model, logger, gpu_id)

    sampler.sample()


def resume_multi_gpu_sampler(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    sampler_params: SamplerParameters,
    data_path: str,
    restart_batch: int,
    resume_save_path: str,
):
    import numpy as np
    from mpi4py import MPI

    mask01, sigma2, obs = load_from_h5(data_path)
    img_size = obs.shape

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    grid_size = np.asarray(MPI.Compute_dims(comm.Get_size(), 2), dtype=int)
    mpi_cart_comm = comm.Create_cart(grid_size)
    ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))

    nb_gpu = cp.cuda.runtime.getDeviceCount()
    gpu_id = rank % nb_gpu

    slicer = CartesianCommSlicer(
        ranknd, grid_size, obs.shape, np.asarray([0, 0]), np.asarray([0, 0])
    )
    tile_size = slicer.tile_size

    with cp.cuda.Device(gpu_id):
        mask = cp.asarray(mask01[slicer.slice_global_buffer_to_tile])
        observations = cp.asarray(obs[slicer.slice_global_buffer_to_tile])

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2)
    X = GpuPSGLA(tile_size, step_size_X, gpu_id)
    Z = GpuPSGLA((2, *tile_size), step_size_Z, gpu_id)

    with cp.cuda.Device(gpu_id):
        model_params = GaussianInpaintingTvParameters(
            observations, mask, sigma2, reg_coef, split_coef
        )

    model = DistributedGaussianInpaintingTvModel(
        comm,
        img_size,
        grid_size,
        model_params,
        X,
        Z,
        gpu_id,
    )

    sampler = MultiGpuSampler(comm, sampler_params, model, logger, gpu_id)

    resume_file_name = join(
        sampler_params.save_path, "sample" + str(restart_batch) + ".h5"
    )
    sampler.restart(resume_file_name, restart_batch, resume_save_path)
    sampler.sample()
