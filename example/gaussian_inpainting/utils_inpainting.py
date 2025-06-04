import json
import logging

import cupy as cp
import numpy as np
from utils_data import load_from_h5

from mcmc.models.gaussian_inpainting_model import (
    DistributedInpaintingModel,
    InpaintingModel,
    InpaintingParameters,
)
from mcmc.operators.masking import Masking
from mcmc.sampler.base_sampler import SamplerParameters
from mcmc.sampler.distributed_sampler import DistributedSampler
from mcmc.sampler.gpu_sampler import GpuSampler
from mcmc.sampler.multi_gpu_sampler import MultiGpuSampler
from mcmc.sampler.serial_sampler import SerialSampler
from mcmc.slicer.cartesian_comm_slicer import CartesianCommSlicer
from mcmc.transition_kernel.gpu_psgla import GpuPSGLA
from mcmc.transition_kernel.psgla import PSGLA
from mcmc.utils.utils import apply_gaussian_noise, generate_observations, load_img_size


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

    model_params = InpaintingParameters(
        observations, mask, sigma2, reg_coef, split_coef
    )

    model = InpaintingModel(model_params, X, Z)

    sampler = SerialSampler(sampler_params, model, logger)

    sampler.sample()


def compute_gpu(
    logger: logging.Logger,
    split_coef: float,
    reg_coef: float,
    sampler_params: SamplerParameters,
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
        data_path, slicer._get_slice_global_buffer_to_tile()
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


def compute_multi_gpu(
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
            data_path, slicer._get_slice_global_buffer_to_tile()
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

    inpainting_operator = Masking(mask)

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
