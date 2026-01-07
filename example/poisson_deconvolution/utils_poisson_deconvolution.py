r"""Utility functions to set the Poisson deconvolution example script for the experiments reported in :cite:p:`Bouton2025` (synthetic data generation, sampling and post-processing steps)."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.

# TODO: revise script to also accommodate gray scale images (for which n_channels = 1 is not explicitly present when using xxx.shape)

import logging
from pathlib import Path

import h5py
import numpy as np

from cards.backend import xp
from cards.models.base_poisson_deconvolution_model import PoissonDeconvolutionParameters
from cards.models.poisson_deconvolution_pnp_model import (
    DistributedPoissonDeconvolutionPnpModel,
    PoissonDeconvolutionPnpModel,
)
from cards.models.poisson_deconvolution_tv_model import (
    DistributedPoissonDeconvolutionTvModel,
    PoissonDeconvolutionTvModel,
)
from cards.operators.dft_convolution import DftConvolution
from cards.operators.mpi_dft_convolution import MpiDftConvolution
from cards.sampler.base_sampler import SamplerParameters
from cards.sampler.distributed_sampler import DistributedSampler
from cards.sampler.gpu_sampler import GpuSampler
from cards.sampler.multi_gpu_sampler import MultiGpuSampler
from cards.sampler.serial_sampler import SerialSampler
from cards.transition_kernel.gpu_pnp_ula import GpuPnpULA
from cards.transition_kernel.gpu_psgla import GpuPSGLA
from cards.transition_kernel.psgla import PSGLA
from cards.utils.path_builder import (
    deconvolution_str,
    obs_dir,
    poisson_str,
)
from cards.utils.utils import extract_subset_from_dict
from cards.utils.utils_img import read_dtype, read_img_shape
from cards.utils.utils_observations import (
    apply_poisson_noise,
    fit_kernel_shape,
    generate_and_save_observations,
    generate_gaussian_kernel,
    generate_motion_kernel,
    slice_linear_conv_to_original,
)


def poisson_deconvolution_params(params: dict) -> dict:
    d = extract_subset_from_dict(params, ["reg_coef", "split_coef1", "split_coef2"])
    if "denoiser_params" in params:
        d.update(extract_subset_from_dict(params, ["denoiser_params"]))
    return d


def define_slices(params: dict) -> dict:
    img_size = read_img_shape(params["original_img_path"])
    # TODO: find a more elegant way to deal with image of different dimensions
    kernel_size = np.asarray([1] * (len(img_size) - 2) + [params["kernel"]["size"]] * 2)
    return {
        "slices": slice_linear_conv_to_original(img_size, kernel_size),
        "dynamic_range": params["dynamic_range"],
    }


def generate_poisson_deconvolution_observations(
    original_img_path: str,
    kernel_params: dict,
    dynamic_range: float,
    data_seed: int,
    obs_path: str,
    maximum: float = 1.0,
):
    gt_size = np.asarray(read_img_shape(original_img_path))
    dtype = read_dtype(original_img_path)

    rng = np.random.default_rng(data_seed)

    if kernel_params["type"] == "motion":
        kernel = generate_motion_kernel(
            kernel_params["size"],
            kernel_params["intensity"],
            dtype,
            rng,
        )
    else:
        kernel = generate_gaussian_kernel(
            kernel_params["size"],
            kernel_params["std"],
            dtype,
        )

    params_saved = {"kernel": kernel}

    reshaped_kernel = fit_kernel_shape(kernel, gt_size)

    obs_dims = gt_size.copy()
    # convolution affects only the last two dimensions (i.e., spatial dimensions)
    obs_dims[-2:] += np.asarray(kernel.shape, dtype=int) - 1

    convolution_handler = DftConvolution(gt_size, reshaped_kernel, obs_dims)

    generate_and_save_observations(
        original_img_path,
        obs_path,
        convolution_handler,
        apply_poisson_noise,
        data_seed,
        params_saved,
        maximum,
        dynamic_range=dynamic_range,
    )


def application_params_dir(params: dict) -> str:
    return f"split1_{params['split_coef1']}-split2_{params['split_coef2']}".replace(
        ".", "_"
    )


def build_obs_and_model_paths(params: dict) -> tuple[str, str]:
    noise_str = poisson_str(params)
    application_str = deconvolution_str(params)
    obs_f = f"poisson-deconvolution/{obs_dir(params, application_str, noise_str)}"
    model_params_f = application_params_dir(params)

    return obs_f, model_params_f


def compute_step_sizes_poisson_deconvolution_tv(
    split_coef1: float,
    split_coef2: float,
    dynamic_range: float,
    kernel: xp.ndarray,
    reg_coef: float,
) -> tuple[float, float, float]:
    """Compute the step sizes for the PSGLA transition kernel with TV prior.

    Parameters
    ----------
    split_coef1 : float
        Splitting coefficient, parameter of the model.
    split_coef2 : float
        Splitting coefficient, parameter of the model.
    dynamic_range : float
        Dynamic range of the observations, related to the poisson noise level.
    kernel : xp.ndarray
        Kernel used in the deconvolution model.

    Returns
    -------
    tuple[float, float]
        Step sizes for the X and Z variables in the PSGLA transition kernel.
    """

    x = 0.99 / (
        dynamic_range**2 / split_coef1 * xp.max(xp.abs(xp.fft.rfft2(kernel))) ** 2
        + 1 / split_coef2
    )
    z1 = 0.99 * split_coef1
    z2 = 0.99 / (reg_coef / split_coef2)

    return x, z1, z2


def compute_step_sizes_poisson_deconvolution_pnp(
    split_coef1: float,
    split_coef2: float,
    dynamic_range: float,
    kernel: xp.ndarray,
    reg_coef: float,
    L: float,
    eps: float,
) -> tuple[float, float, float, float]:
    x = 0.99 / (
        dynamic_range**2 / split_coef1 * xp.max(xp.abs(xp.fft.rfft2(kernel))) ** 2
        + 1 / split_coef2
    )
    z1 = 0.99 * split_coef1
    Ly = 0.99 * split_coef2

    lambda_ = 0.99 / (2 * L / eps + 4 * Ly)
    be = (reg_coef * L) / eps + 1 / lambda_ + Ly
    z2 = 0.99 / (3 * be)
    return x, z1, z2, lambda_


def load_from_h5(
    filename: str | Path,
) -> tuple[xp.ndarray, float, tuple[int, ...], tuple[int, ...]]:
    """Load the kernel, dynamic_range, and observation shape from an HDF5 file.

    Parameters
    ----------
    filename : str or Path
        Path to the HDF5 file containing the kernel, dynamic_range, and observations.

    Returns
    -------
    tuple[xp.ndarray, float, tuple[int, ...], tuple[int, ...]]
        A tuple containing the kernel as a cupy array,
        the dynamic_range as a float,
        the shape of the ground truth as a tuple of integers,
        and the shape of the observations as a tuple of integers.
    """
    with h5py.File(filename, "r") as data_file:
        kernel = xp.asarray(data_file["kernel"])
        dynamic_range = data_file["dynamic_range"][()]  # type: ignore
        gt_shape = data_file["x"].shape  # type: ignore
        obs_shape = data_file["y"].shape  # type: ignore
    return kernel, dynamic_range, gt_shape, obs_shape  # type: ignore


def compute_tv(
    logger: logging.Logger,
    sampler_params: SamplerParameters,
    obs_path: str,
    reg_coef: float,
    split_coef1: float,
    split_coef2: float,
    mode: str = "serial",
    device: str = "cpu",
):
    kernel, dynamic_range, gt_shape, _ = load_from_h5(obs_path)
    step_size_X, step_size_Z1, step_size_Z2 = (
        compute_step_sizes_poisson_deconvolution_tv(
            split_coef1,
            split_coef2,
            dynamic_range,
            kernel,
            reg_coef,
        )
    )

    kernel = fit_kernel_shape(kernel, gt_shape)

    match mode:
        case "mpi":
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            # MPI.Compute_dims(size, 2)
            grid_size = np.asarray([1] * (len(gt_shape) - 2) + [size, 1])

            op = MpiDftConvolution(np.asarray(gt_shape), kernel, comm, grid_size)
            y = np.empty(
                op.adjoint_communicator.cartslicer.tile_size, dtype=kernel.dtype
            )
            with h5py.File(obs_path, "r", driver="mpio", comm=comm) as f:
                f["y"].read_direct(  # type: ignore
                    y,
                    op.adjoint_communicator.cartslicer.slice_global_buffer_to_tile,
                )
            if "gpu" in device:
                y = xp.asarray(y)

            state_shape = tuple(op.direct_communicator.cartslicer.tile_size)
        case "serial":
            with h5py.File(obs_path, "r") as f:
                y = xp.asarray(f["y"])
            state_shape = gt_shape
        case _:
            raise ValueError(f"Unknown run mode: {mode}")

    model_params = PoissonDeconvolutionParameters(
        y,
        kernel,
        dynamic_range,
        reg_coef,
        split_coef1,
        split_coef2,
    )

    match device:
        case "cpu":
            X = PSGLA(state_shape, step_size_X, dtype=y.dtype)
            Z1 = PSGLA(y.shape, step_size_Z1, dtype=y.dtype)
            Z2 = PSGLA((2, *state_shape), step_size_Z2, dtype=y.dtype)
        case "gpu":
            X = GpuPSGLA(state_shape, step_size_X, dtype=y.dtype)
            Z1 = GpuPSGLA(y.shape, step_size_Z1, dtype=y.dtype)
            Z2 = GpuPSGLA((2, *state_shape), step_size_Z2, dtype=y.dtype)
        case _:
            raise ValueError(f"Unknown device: {device}")

    match mode:
        case "mpi":
            model = DistributedPoissonDeconvolutionTvModel(
                comm,
                np.asarray(gt_shape),
                grid_size,
                model_params,
                X,
                Z1,
                Z2,
            )

            if device == "cpu":
                sampler = DistributedSampler(comm, sampler_params, model, logger)
            else:
                # TODO: revise / generalise default gpu assignment
                sampler = MultiGpuSampler(
                    comm,
                    sampler_params,
                    model,
                    logger,
                    gpu_id=comm.Get_rank() % xp.cuda.runtime.getDeviceCount(),
                )
        case "serial":
            model = PoissonDeconvolutionTvModel(model_params, X, Z1, Z2)
            Sampler = SerialSampler if device == "cpu" else GpuSampler
            sampler = Sampler(sampler_params, model, logger)
        case _:
            raise ValueError(f"Unknown run mode: {mode}")
    sampler.sample()


def compute_pnp(
    logger: logging.Logger,
    sampler_params: SamplerParameters,
    obs_path: str,
    reg_coef: float,
    split_coef1: float,
    split_coef2: float,
    denoiser_params: dict,
    mode: str = "serial",
    device: str = "cpu",
):
    eps = denoiser_params["denoising_level"] ** 2
    kernel, dynamic_range, gt_shape, _ = load_from_h5(obs_path)

    step_size_X, step_size_Z1, step_size_Z2, lambda_ = (
        compute_step_sizes_poisson_deconvolution_pnp(
            split_coef1,
            split_coef2,
            dynamic_range,
            kernel,
            reg_coef,
            L=denoiser_params.get("L", None) or 1.0,
            eps=eps,
        )
    )

    kernel = fit_kernel_shape(kernel, gt_shape)

    match mode:
        case "mpi":
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            grid_size = np.asarray([1] * (len(gt_shape) - 2) + [size, 1])

            op = MpiDftConvolution(np.asarray(gt_shape), kernel, comm, grid_size)
            y = np.empty(
                op.adjoint_communicator.cartslicer.tile_size, dtype=kernel.dtype
            )
            with h5py.File(obs_path, "r", driver="mpio", comm=comm) as f:
                f["y"].read_direct(  # type: ignore
                    y,
                    op.adjoint_communicator.cartslicer.slice_global_buffer_to_tile,
                )
            if "gpu" in device:
                y = xp.asarray(y)

            state_shape = tuple(op.direct_communicator.cartslicer.tile_size)
        case "serial":
            with h5py.File(obs_path, "r") as f:
                y = xp.asarray(f["y"])
            state_shape = gt_shape
        case _:
            raise ValueError(f"Unknown run mode: {mode}")

    model_params = PoissonDeconvolutionParameters(
        y,
        kernel,
        dynamic_range,
        reg_coef,
        split_coef1,
        split_coef2,
    )

    match device:
        case "cpu":
            raise NotImplementedError("PnP is not implemented for CPU device.")
        case "gpu":
            X = GpuPSGLA(state_shape, step_size_X, dtype=y.dtype)
            Z1 = GpuPSGLA(y.shape, step_size_Z1, dtype=y.dtype)
            Z2 = GpuPnpULA(
                state_shape, step_size_Z2, reg_coef, eps, lambda_, dtype=y.dtype
            )
        case _:
            raise ValueError(f"Unknown device: {device}")

    match mode:
        case "mpi":
            match denoiser_params["type"]:
                case "ddfb":
                    from cards.denoisers.mpi_ddfb import MpiDDFB

                    denoiser = MpiDDFB(
                        comm,
                        grid_size,
                        image_size=np.asarray(gt_shape),
                        n_layers=denoiser_params["n_layers"],
                        n_features=denoiser_params["n_features"],
                    )
                case "dncnn":
                    from cards.denoisers.mpi_dncnn import MpiDnCNN

                    denoiser = MpiDnCNN(
                        comm, grid_size, image_size=np.asarray(gt_shape)
                    )
                case _:
                    raise ValueError(
                        f"Unknown denoiser type: {denoiser_params['type']}"
                    )
            model = DistributedPoissonDeconvolutionPnpModel(
                comm,
                np.asarray(gt_shape),
                grid_size,
                model_params,
                X,
                Z1,
                Z2,
                denoiser,
            )
            if device == "cpu":
                sampler = DistributedSampler(comm, sampler_params, model, logger)
            else:
                # TODO: revise / generalise default gpu assignment
                sampler = MultiGpuSampler(
                    comm,
                    sampler_params,
                    model,
                    logger,
                    comm.Get_rank() % xp.cuda.runtime.getDeviceCount(),
                )
        case "serial":
            match denoiser_params["type"]:
                case "ddfb":
                    from cards.denoisers.serial_ddfb import SerialDDFB

                    denoiser = SerialDDFB(
                        image_size=np.asarray(gt_shape),
                        n_layers=denoiser_params["n_layers"],
                        n_features=denoiser_params["n_features"],
                    )
                case "dncnn":
                    from cards.denoisers.serial_dncnn import SerialDnCNN

                    denoiser = SerialDnCNN(image_size=np.asarray(gt_shape))
                case "drunet":
                    from cards.denoisers.serial_drunet import SerialDRUNet

                    denoiser = SerialDRUNet(image_size=np.asarray(gt_shape))
                case _:
                    raise ValueError(
                        f"Unknown denoiser type: {denoiser_params['type']}"
                    )
            model = PoissonDeconvolutionPnpModel(model_params, X, Z1, Z2, denoiser)
            Sampler = SerialSampler if device == "cpu" else GpuSampler
            sampler = Sampler(sampler_params, model, logger)
        case _:
            raise ValueError(f"Unknown run mode: {mode}")

    sampler.sample()
