import logging
from os.path import join

import cupy as cp
import h5py
from cards.models.gaussian_inpainting_model import (
    DistributedInpaintingModel,
    InpaintingModel,
    InpaintingParameters,
)

from cards.sampler.base_sampler import SamplerParameters
from cards.sampler.distributed_sampler import DistributedSampler
from cards.sampler.gpu_sampler import GpuSampler
from cards.sampler.multi_gpu_sampler import MultiGpuSampler
from cards.sampler.serial_sampler import SerialSampler
from cards.slicer.cartesian_comm_slicer import CartesianCommSlicer
from cards.transition_kernel.gpu_psgla import GpuPSGLA
from cards.transition_kernel.psgla import PSGLA
from cards.utils.utils import load_img_size


def load_from_h5(filename, local_slice=slice(None)):
    """load the mask01, sig2 and data entries from the h5 file"""
    with h5py.File(filename, "r") as data_file:
        mask = data_file["mask01"][local_slice]
        sigma2 = data_file["sigma2"][()]
        observations = data_file["data"][local_slice]
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

    model_params = InpaintingParameters(
        observations, mask, sigma2, reg_coef, split_coef
    )

    model = InpaintingModel(
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

    model_params = InpaintingParameters(
        observations, mask, sigma2, reg_coef, split_coef
    )

    model = InpaintingModel(
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

    model_params = InpaintingParameters(
        cp.asarray(observations), cp.asarray(mask), sigma2, reg_coef, split_coef
    )

    model = InpaintingModel(
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

    model_params = InpaintingParameters(
        cp.asarray(observations), cp.asarray(mask), sigma2, reg_coef, split_coef
    )

    model = InpaintingModel(
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
    r"""compute_distributed Generate a sample with the distributed implementation.

    Parameters
    ----------
    logger: logging.Logger
        Logger object.
    sampler_params : SamplerParameters
        Parameters used by the sampler.
    seed : int
        Seed.
    save_path : str
        Path to the directory where we will save the sample.
    data_path : str
        Path to the file containing the deteriorated signal.
    """
    import numpy as np
    from mpi4py import MPI

    img_size = load_img_size(data_path)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    grid_size = np.asarray(MPI.Compute_dims(comm.Get_size(), 2), dtype=int)
    mpi_cart_comm = comm.Create_cart(grid_size)
    ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))

    slicer = CartesianCommSlicer(
        ranknd, grid_size, img_size, np.asarray([0, 0]), np.asarray([0, 0])
    )

    mask, sigma2, observations = load_from_h5(
        data_path, slicer.slice_global_buffer_to_tile
    )

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2)
    tile_size = slicer.tile_size
    X = PSGLA(tile_size, step_size_X)
    Z = PSGLA((2, *tile_size), step_size_Z)

    model_params = InpaintingParameters(
        observations, mask, sigma2, reg_coef, split_coef
    )

    model = DistributedInpaintingModel(
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

    img_size = load_img_size(data_path)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    grid_size = np.asarray(MPI.Compute_dims(comm.Get_size(), 2), dtype=int)
    mpi_cart_comm = comm.Create_cart(grid_size)
    ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))

    slicer = CartesianCommSlicer(
        ranknd, grid_size, img_size, np.asarray([0, 0]), np.asarray([0, 0])
    )

    mask, sigma2, observations = load_from_h5(
        data_path, slicer.slice_global_buffer_to_tile
    )

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2)
    tile_size = slicer.tile_size
    X = PSGLA(tile_size, step_size_X)
    Z = PSGLA((2, *tile_size), step_size_Z)

    model_params = InpaintingParameters(
        observations, mask, sigma2, reg_coef, split_coef
    )

    model = DistributedInpaintingModel(
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

    img_size = load_img_size(data_path)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    grid_size = np.asarray(MPI.Compute_dims(comm.Get_size(), 2), dtype=int)
    mpi_cart_comm = comm.Create_cart(grid_size)
    ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))
    nb_gpu = cp.cuda.runtime.getDeviceCount()
    gpu_id = rank % nb_gpu

    slicer = CartesianCommSlicer(
        ranknd, grid_size, img_size, np.asarray([0, 0]), np.asarray([0, 0])
    )
    tile_size = slicer.tile_size

    with cp.cuda.Device(gpu_id):
        mask01, sigma2, obs = load_from_h5(
            data_path, slicer.slice_global_buffer_to_tile
        )
        mask = cp.asarray(mask01)
        observations = cp.asarray(obs)

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2)

    X = GpuPSGLA(tile_size, step_size_X, gpu_id)
    Z = GpuPSGLA((2, *tile_size), step_size_Z, gpu_id)

    with cp.cuda.Device(gpu_id):
        model_params = InpaintingParameters(
            observations, mask, sigma2, reg_coef, split_coef
        )

    model = DistributedInpaintingModel(
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

    img_size = load_img_size(data_path)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    grid_size = np.asarray(MPI.Compute_dims(comm.Get_size(), 2), dtype=int)
    mpi_cart_comm = comm.Create_cart(grid_size)
    ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))
    nb_gpu = cp.cuda.runtime.getDeviceCount()
    gpu_id = rank % nb_gpu

    slicer = CartesianCommSlicer(
        ranknd, grid_size, img_size, np.asarray([0, 0]), np.asarray([0, 0])
    )
    tile_size = slicer.tile_size

    with cp.cuda.Device(gpu_id):
        mask01, sigma2, obs = load_from_h5(
            data_path, slicer.slice_global_buffer_to_tile
        )
        mask = cp.asarray(mask01)
        observations = cp.asarray(obs)

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2)

    X = GpuPSGLA(tile_size, step_size_X, gpu_id)
    Z = GpuPSGLA((2, *tile_size), step_size_Z, gpu_id)

    with cp.cuda.Device(gpu_id):
        model_params = InpaintingParameters(
            observations, mask, sigma2, reg_coef, split_coef
        )

    model = DistributedInpaintingModel(
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
