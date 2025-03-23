import json
import logging

import cupy as cp
import h5py
import numpy as np
import scipy

from mcmc.models.GaussianDeconvolutionModel import GaussianDeconvolutionModel
from mcmc.models.GpuGaussianDeconvolutionModel import GpuGaussianDeconvolutionModel
from mcmc.models.DistributedGaussianDeconvolutionModel import (
    DistributedGaussianDeconvolutionModel,
)
from mcmc.models.MultiGpuGaussianDeconvolutionModel import (
    MultiGpuGaussianDeconvolutionModel,
)
from mcmc.sampler.GpuSampler import GpuSampler
from mcmc.sampler.SerialSampler import Sampler
from mcmc.sampler.DistributedSampler import DistributedSampler
from mcmc.sampler.MultiGpuSampler import MultiGpuSampler
from mcmc.TransitionKernel.GpuTransitionKernel import GpuPSGLA, MultiGpuPSGLA
from mcmc.TransitionKernel.TransitionKernel import PSGLA
from mcmc.operators.serial_convolution import SerialConvolution
from mcmc.slicer.cartesian_comm_slicer import CartesianCommSlicer
from mcmc.utils.utils import load_img_size, generate_observations, apply_gaussian_noise
from mcmc.distributed_operators.sync_linear_convolution import SyncLinearConvolution
from mcmc.distributed_operators.multi_gpu.dft_convolution import MultiGPU_DFTConvolution


def slice_obs_to_original(img_dims, kernel_dims):
    s = tuple(
        [
            np.s_[kernel_dims[d] // 2 : img_dims[d] + kernel_dims[d] // 2]
            for d in range(len(img_dims))
        ]
    )
    return s


def load_from_h5(filename):
    """load the mask01, sig2 and data entries from the h5 file"""
    with h5py.File(filename, "r") as data_file:
        kernel = data_file["kernel"][:]
        sigma2 = data_file["sigma2"][()]
        observations = data_file["data"][:]
    return kernel, sigma2, observations


def add_deconvolution_param(config_file_path: str, args: dict) -> None:
    config_file = open(config_file_path)
    params = json.load(config_file)

    args["split_coef"] = params["alpha"]
    args["reg_coef"] = params["regularizationCoefficient"]
    return


def compute_step_size(split_coef: float, sigma2: float, kernel: np.ndarray):
    x = 0.99 / (8.0 / split_coef + np.max(np.abs(np.fft.rfft2(kernel))) ** 2 / sigma2)
    z = 0.99 * split_coef
    return (x, z)


def generate_gaussian_kernel(kernel_size, kernel_std) -> np.ndarray:
    r"""Generate a square normalized 2D Gaussian kernel.

    Parameters
    ----------
    kernel_size : int
        Size of one dimension of the kernel.
    kernel_std : float
        Standard deviation of the Gaussian kernel.

    Note
    ----
    Equivalent to the ``fspecial('gaussian', ...)`` function in Matlab.

    Returns
    -------
    h : numpy.ndarray
        Square Gaussian kernel with :math:`\|h\|_1 = 1`.
    """
    # equivalent to fspecial('gaussian', ...) in Matlab
    w = scipy.signal.windows.gaussian(kernel_size, kernel_std)
    h = w[:, np.newaxis] * w[np.newaxis, :]
    h = h / np.sum(h)
    return h


def generate_gaussian_deconvolution_observations(
    original_path: str,
    kernel_dims: np.ndarray,
    kernel_std: float,
    snr: float,
    data_seed: int,
    obs_path: str,
    maximum: float = 1.0,
):
    img_dims = load_img_size(original_path)
    kernel = generate_gaussian_kernel(kernel_dims, kernel_std)

    obs_dims = np.asarray(img_dims) + np.asarray(kernel_dims) - np.ones_like(img_dims)
    convolution_handler = SerialConvolution(np.asarray(img_dims), kernel, obs_dims)

    pb_params = {}
    pb_params["kernel"] = kernel

    generate_observations(
        original_path,
        convolution_handler,
        snr,
        apply_gaussian_noise,
        data_seed,
        obs_path,
        maximum=1.0,
        problem_parameters=pb_params,
    )

    return


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
    r"""compute_serial Generate a sample withe the serial implementation.

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

    kernel, sigma2, observations = load_from_h5(data_path)

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2, kernel)

    Mm, Nn = observations.shape
    m, n = kernel.shape
    M = Mm - m + 1
    N = Nn - n + 1
    X = PSGLA(np.asarray([M, N], dtype=int), step_size_X)
    Z = PSGLA((2, *X.current_state.shape), step_size_Z)

    model = GaussianDeconvolutionModel(
        observations, kernel, X, Z, sigma2, reg_coef, split_coef
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
    r"""compute_serial Generate a sample withe the serial implementation.

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

    kernel, sigma2, observations = load_from_h5(data_path)

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2, cp.asnumpy(kernel)) #! giving cp array to numpy function returns cp array

    Mm, Nn = observations.shape
    m, n = kernel.shape
    M = Mm - m + 1
    N = Nn - n + 1
    X = GpuPSGLA(np.asarray([M, N], dtype=int), step_size_X)
    Z = GpuPSGLA((2, *X.current_state.shape), step_size_Z)

    model = GpuGaussianDeconvolutionModel(
        cp.asarray(observations), cp.asarray(kernel), X, Z, sigma2, reg_coef, split_coef
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

    kernel, sigma2, observations = load_from_h5(data_path)
    Mm, Nn = observations.shape
    m, n = kernel.shape
    M = Mm - m + 1
    N = Nn - n + 1
    img_size = np.asarray([M, N])

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

    convolution = SyncLinearConvolution(img_size, kernel, comm, grid_size)

    observations = observations[
        convolution.adjoint_communicator.cartslicer.slice_global_buffer_to_tile
    ]

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2, kernel)
    X = PSGLA(tile_size, step_size_X)
    Z = PSGLA((2, *tile_size), step_size_Z)

    model = DistributedGaussianDeconvolutionModel(
        comm,
        img_size,
        grid_size,
        observations,
        kernel,
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
    r"""compute_multi_gpu Generate a sample with the distributed implementation on gpu.

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

    kernel, sigma2, observations = load_from_h5(data_path)
    Mm, Nn = observations.shape
    m, n = kernel.shape
    M = Mm - m + 1
    N = Nn - n + 1
    img_size = np.asarray([M, N])

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

    convolution = MultiGPU_DFTConvolution(
        img_size, cp.asarray(kernel), comm, grid_size
    )  # compute dimensions another way, waste of memory

    with cp.cuda.Device(gpu_id):
        observations = cp.asarray(
            observations[
                convolution.adjoint_communicator.cartslicer.slice_global_buffer_to_tile
            ]
        )
        kernel = cp.asarray(kernel)

    step_size_X, step_size_Z = compute_step_size(split_coef, sigma2, cp.asnumpy(kernel))
    X = MultiGpuPSGLA(tile_size, step_size_X, gpu_id)
    Z = MultiGpuPSGLA((2, *tile_size), step_size_Z, gpu_id)

    model = MultiGpuGaussianDeconvolutionModel(
        comm,
        img_size,
        grid_size,
        observations,
        kernel,
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
