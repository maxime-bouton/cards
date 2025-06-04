import logging

import cupy as cp
import numpy as np
from utils_data import load_from_h5, load_sizes_from_h5

from mcmc.models.gaussian_deconvolution_model import (
    DeconvolutionParameters,
    DistributedGaussianDeconvolutionModel,
    GaussianDeconvolutionModel,
)
from mcmc.sampler.base_sampler import SamplerParameters
from mcmc.sampler.distributed_sampler import DistributedSampler
from mcmc.sampler.gpu_sampler import GpuSampler
from mcmc.sampler.multi_gpu_sampler import MultiGpuSampler
from mcmc.sampler.serial_sampler import SerialSampler
from mcmc.slicer.cartesian_comm_slicer import CartesianCommSlicer
from mcmc.transition_kernel.gpu_psgla import GpuPSGLA
from mcmc.transition_kernel.psgla import PSGLA


def compute_step_size(split_coef: float, sigma2: float, kernel: np.ndarray):
    x = 0.99 / (8.0 / split_coef + np.max(np.abs(np.fft.rfft2(kernel))) ** 2 / sigma2)
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
    r"""compute_serial Generate a sample withe the serial implementation.

    Parameters
    ----------
    logger: logging.Logger
        Logger object.
    split_coef : float
        Splitting coefficient, parameter of the model.
    reg_coef : float
        Regularization coefficient, parameter of the model.
    sampler_params : SamplerParameters
        Dataclass containing the parameters used by the sampler.
    data_path : str
        Path to the file containing the deteriorated signal.
    """

    kernel, sigma2, observations = load_from_h5(data_path)

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2, kernel)

    Mm, Nn = observations.shape
    m, n = kernel.shape
    M = Mm - m + 1
    N = Nn - n + 1
    X = PSGLA(np.asarray([M, N], dtype=int), step_size_X)
    Z = PSGLA((2, *X.current_state.shape), step_size_Z)

    model_params = DeconvolutionParameters(
        observations, kernel, sigma2, reg_coef, split_coef
    )

    model = GaussianDeconvolutionModel(
        model_params,
        X,
        Z,
    )

    sampler = SerialSampler(sampler_params, model, logger)

    sampler.sample()


def compute_gpu(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    sampler_params: SamplerParameters,
    data_path: str,
):
    r"""compute_serial Generate a sample withe the serial implementation.

    Parameters
    ----------
    logger: logging.Logger
        Logger object.
    split_coef : float
        Splitting coefficient, parameter of the model.
    reg_coef : float
        Regularization coefficient, parameter of the model.
    sampler_params : SamplerParameters
        Dataclass containing the parameters used by the sampler.
    data_path : str
        Path to the file containing the deteriorated signal.
    """

    kernel, sigma2, observations = load_from_h5(data_path)

    step_size_X, step_size_Z = compute_step_size(
        split_coef, sigma2, cp.asnumpy(kernel)
    )  #! giving cp array to numpy function returns cp array

    Mm, Nn = observations.shape
    m, n = kernel.shape
    M = Mm - m + 1
    N = Nn - n + 1
    X = GpuPSGLA(np.asarray([M, N], dtype=int), step_size_X)
    Z = GpuPSGLA((2, *X.current_state.shape), step_size_Z)

    model_params = DeconvolutionParameters(
        cp.asarray(observations), cp.asarray(kernel), sigma2, reg_coef, split_coef
    )

    model = GaussianDeconvolutionModel(
        model_params,
        X,
        Z,
    )

    sampler = GpuSampler(sampler_params, model, logger)

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
    split_coef : float
        Splitting coefficient, parameter of the model.
    reg_coef : float
        Regularization coefficient, parameter of the model.
    sampler_params : SamplerParameters
        Dataclass containing the parameters used by the sampler.
    data_path : str
        Path to the file containing the deteriorated signal.
    """
    import numpy as np
    from mpi4py import MPI

    # TODO: also for multi-GPU: do not read the full observation array (only read a slice) -> extract dataset sizes to create slices -> DataManager?(different behaviour?)
    # TODO: - read dataset size for observations
    #       - read paramaters needed in all workers
    #       - create slices to read required observations
    obs_shape, kernel_shape = load_sizes_from_h5(data_path)
    img_size = np.asarray(obs_shape) - np.asarray(kernel_shape) + 1

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    grid_size = np.asarray(MPI.Compute_dims(comm.Get_size(), 2), dtype=int)
    mpi_cart_comm = comm.Create_cart(grid_size)
    ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))

    slicer = CartesianCommSlicer(
        ranknd,
        grid_size,
        img_size,
        np.asarray([0, 0]),
        np.asarray([0, 0]),
    )
    tile_size = slicer.tile_size

    # build local slice, copied from slicer
    overlap_size = np.array(kernel_shape, dtype="i") - 1
    local_data_size = slicer.tile_size + (ranknd == 0) * overlap_size
    offset_id = (ranknd > 0) * overlap_size

    tile_data = np.zeros((img_size.size, 2), dtype="i")
    tile_data[:, 0] = slicer.tile_range[:, 0] + offset_id
    tile_data[:, 1] = tile_data[:, 0] + local_data_size - 1

    slice_obs = tuple([np.s_[tile_data[d, 0] : tile_data[d, 1] + 1] for d in range(2)])

    kernel, sigma2, observations = load_from_h5(data_path, slice_obs)

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2, kernel)
    X = PSGLA(tile_size, step_size_X)
    Z = PSGLA((2, *tile_size), step_size_Z)

    model_params = DeconvolutionParameters(
        observations, kernel, sigma2, reg_coef, split_coef
    )

    model = DistributedGaussianDeconvolutionModel(
        comm,
        img_size,
        grid_size,
        model_params,
        X,
        Z,
    )

    sampler = DistributedSampler(comm, sampler_params, model, logger)

    sampler.sample()


def compute_multi_gpu(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    sampler_params: SamplerParameters,
    data_path: str,
):
    r"""compute_multi_gpu Generate a sample with the distributed implementation on gpu.

    Parameters
    ----------
    logger: logging.Logger
        Logger object.
    split_coef : float
        Splitting coefficient, parameter of the model.
    reg_coef : float
        Regularization coefficient, parameter of the model.
    sampler_params : SamplerParameters
        Dataclass containing the parameters used by the sampler.
    data_path : str
        Path to the file containing the deteriorated signal.
    """
    import numpy as np
    from mpi4py import MPI

    obs_shape, kernel_shape = load_sizes_from_h5(data_path)
    img_size = np.asarray(obs_shape) - np.asarray(kernel_shape) + 1

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    grid_size = np.asarray(MPI.Compute_dims(comm.Get_size(), 2), dtype=int)
    mpi_cart_comm = comm.Create_cart(grid_size)
    ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))
    nb_gpu = cp.cuda.runtime.getDeviceCount()
    gpu_id = rank % nb_gpu

    slicer = CartesianCommSlicer(
        ranknd,
        grid_size,
        img_size,
        np.asarray([0, 0]),
        np.asarray([0, 0]),
    )
    tile_size = slicer.tile_size

    # build local slice, copied from slicer
    overlap_size = np.array(kernel_shape, dtype="i") - 1
    local_data_size = slicer.tile_size + (ranknd == 0) * overlap_size
    offset_id = (ranknd > 0) * overlap_size

    tile_data = np.zeros((img_size.size, 2), dtype="i")
    tile_data[:, 0] = slicer.tile_range[:, 0] + offset_id
    tile_data[:, 1] = tile_data[:, 0] + local_data_size - 1

    slice_obs = tuple([np.s_[tile_data[d, 0] : tile_data[d, 1] + 1] for d in range(2)])

    kernel, sigma2, observations = load_from_h5(data_path, slice_obs)

    with cp.cuda.Device(gpu_id):
        observations = cp.asarray(observations)
        kernel = cp.asarray(kernel)

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2, cp.asnumpy(kernel))
    X = GpuPSGLA(tile_size, step_size_X, gpu_id)
    Z = GpuPSGLA((2, *tile_size), step_size_Z, gpu_id)

    with cp.cuda.Device(gpu_id):
        model_params = DeconvolutionParameters(
            observations, kernel, sigma2, reg_coef, split_coef
        )

    model = DistributedGaussianDeconvolutionModel(
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
