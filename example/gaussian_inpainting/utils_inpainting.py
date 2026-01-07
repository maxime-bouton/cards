r"""Utility functions to set the Gaussian inpainting example script for the experiments reported in :cite:p:`Bouton2025` (synthetic data generation, sampling and post-processing steps)."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.

# TODO: revise script to also accommodate gray scale images (for which n_channels = 1 is not explicitly present when using xxx.shape)

import logging

import h5py
import numpy as np
from scipy import interpolate

from cards.backend import xp
from cards.models.gaussian_inpainting_pnp_model import (
    DistributedGaussianInpaintingPnpModel,
    GaussianInpaintingPnpModel,
    GaussianInpaintingPnpParameters,
)
from cards.models.gaussian_inpainting_tv_model import (
    DistributedGaussianInpaintingTvModel,
    GaussianInpaintingTvModel,
    GaussianInpaintingTvParameters,
)
from cards.operators.masking import Masking
from cards.sampler.base_sampler import SamplerParameters
from cards.sampler.distributed_sampler import DistributedSampler
from cards.sampler.gpu_sampler import GpuSampler
from cards.sampler.multi_gpu_sampler import MultiGpuSampler
from cards.sampler.serial_sampler import SerialSampler
from cards.slicer.cartesian_comm_slicer import CartesianCommSlicer
from cards.transition_kernel.gpu_pnp_ula import GpuPnpULA
from cards.transition_kernel.gpu_psgla import GpuPSGLA
from cards.transition_kernel.psgla import PSGLA
from cards.utils.path_builder import gaussian_str, inpainting_str, obs_dir
from cards.utils.utils import extract_subset_from_dict
from cards.utils.utils_img import load_img, read_dtype, read_img_shape
from cards.utils.utils_observations import (
    apply_target_gaussian_noise,
    fit_mask_shape,
    generate_and_save_observations,
)


def gaussian_inpainting_params(params: dict) -> dict:
    if "denoiser_params" in params:
        return extract_subset_from_dict(params, ["reg_coef", "denoiser_params"])
    else:
        return extract_subset_from_dict(params, ["reg_coef", "split_coef"])


def interpolate_masked_image_cubic(
    masked_image: xp.ndarray,
    mask: xp.ndarray,
) -> xp.ndarray:
    """
    Interpolate masked values in an image using cubic spline interpolation.
    Transfers data to CPU for interpolation.

    Parameters
    ----------
    masked_image : xp.ndarray
        Image with masked values, shape (C, H, W)
    mask : xp.ndarray
        Boolean mask where True/1 indicates visible pixels, shape (C, H, W)

    Returns
    -------
    xp.ndarray
        Interpolated image with the same shape as the input
    """

    # NOTE: ensure gray image and associated mask have at least 3 axis to reuse
    # the same code for gray and color images
    if len(masked_image.shape) < 3:
        masked_image = masked_image[None, ...]
        mask = mask[None, ...]
    C, H, W = masked_image.shape
    result = masked_image.copy()
    C_mask = mask.shape[-3]
    for c in range(C):
        channel_gpu = masked_image[c]
        mask_gpu = xp.asarray(mask[min(c, C_mask - 1)]).astype(bool)

        if xp.all(~mask_gpu) or xp.all(mask_gpu):
            continue

        # TODO: simplify below, channel_cpu and mask_cpu should not be needed
        if xp.__name__ == "cupy":
            channel_cpu = channel_gpu.get()
            mask_cpu = mask_gpu.get()
        else:
            channel_cpu = channel_gpu
            mask_cpu = mask_gpu

        known_coords = np.where(mask_cpu)
        known_values = channel_cpu[known_coords]

        y_grid, x_grid = np.mgrid[0:H, 0:W]

        filled_channel = interpolate.griddata(
            np.column_stack((known_coords[0], known_coords[1])),
            known_values,
            (y_grid, x_grid),
            method="cubic",
            fill_value=np.mean(known_values),
        )

        filled_channel_gpu = xp.asarray(filled_channel)
        result[c][~mask_gpu] = filled_channel_gpu[~mask_gpu]

    return result


def generate_interpolation(obs_path, mask):
    y = load_img(obs_path, key="y")
    interpolated_y = interpolate_masked_image_cubic(y, mask).clip(0, 1)

    with h5py.File(obs_path, "a") as file:
        file["interpolation"] = (
            interpolated_y
            if isinstance(interpolated_y, np.ndarray)
            else interpolated_y.get()
        )


def generate_inpainting_observations(
    original_img_path: str,
    mask_loss: float,
    isnr: float,
    data_seed: int,
    obs_path: str,
    maximum: float = 1.0,
):
    gt_size = np.asarray(read_img_shape(original_img_path))
    rng = np.random.default_rng(data_seed)
    mask = rng.random(gt_size[-2:]) < (1 - mask_loss)
    inpainting_params = {"mask": mask.copy()}

    mask_extended = fit_mask_shape(xp.asarray(mask), gt_size)
    inpainting_operator = Masking(mask_extended)

    generate_and_save_observations(
        original_img_path,
        obs_path,
        inpainting_operator,
        apply_target_gaussian_noise,
        data_seed,
        inpainting_params,
        maximum,
        isnr=isnr,
    )
    generate_interpolation(obs_path, mask_extended)


def application_params_dir(params: dict) -> str:
    if "denoiser_params" in params:
        return "."
    return f"split{params['split_coef']}".replace(".", "_")


def build_obs_and_model_paths(params: dict) -> tuple[str, str]:
    noise_str = gaussian_str(params)
    application_str = inpainting_str(params)
    obs_f = f"gaussian-inpainting/{obs_dir(params, application_str, noise_str)}"
    model_params_f = application_params_dir(params)

    return obs_f, model_params_f


def compute_step_sizes_gaussian_inpainting_tv(
    split_coef: float,
    sigma2: float,
) -> tuple[float, float]:
    x = 0.99 * 1.0 / (8.0 / split_coef + 1.0 / sigma2)
    z = 0.99 * split_coef
    return x, z


def compute_step_sizes_gaussian_inpainting_pnp(
    sigma2: float,
    reg_coef: float,
    L: float,
    eps: float | None = None,
) -> tuple[float, float]:
    eps = eps or sigma2
    Ly = 1 / sigma2
    lambda_ = 0.99 / (2 * L / eps + 4 * Ly)
    be = (reg_coef * L) / eps + 1 / lambda_ + Ly
    step_size_X = 0.99 / (3 * be)
    return step_size_X, lambda_


def load_from_h5(filename) -> tuple[xp.ndarray, float, tuple[int, ...]]:
    with h5py.File(filename, "r") as data_file:
        mask = xp.asarray(data_file["mask"])
        sigma2 = data_file["sigma2"][()]  # type: ignore
        gt_shape = data_file["x"].shape  # type: ignore
    return mask, sigma2, gt_shape


def compute_tv(
    logger: logging.Logger,
    sampler_params: SamplerParameters,
    obs_path: str,
    reg_coef: float,
    split_coef: float,
    mode: str = "serial",
    device: str = "cpu",
):
    # FIXME: mask is loaded entirely on all processes in MPI mode
    mask, sigma2, gt_shape = load_from_h5(obs_path)
    step_size_X, step_size_Z = compute_step_sizes_gaussian_inpainting_tv(
        split_coef, sigma2
    )

    mask = fit_mask_shape(mask, gt_shape)

    match mode:
        case "mpi":
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()
            # MPI.Compute_dims(size, 2)
            grid_size = np.asarray([1] * (len(gt_shape) - 2) + [size, 1])
            mpi_cart_comm = comm.Create_cart(grid_size)
            ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))

            cartslicer = CartesianCommSlicer(
                ranknd,
                grid_size,
                gt_shape,
                np.zeros(len(grid_size), dtype=int),
                np.zeros(len(grid_size), dtype=int),
            )
            state_shape = tuple(cartslicer.tile_size)

            mask = mask[cartslicer.slice_global_buffer_to_tile]

            dtype = read_dtype(obs_path, "y")
            y = np.empty(state_shape, dtype=dtype)
            with h5py.File(obs_path, "r", driver="mpio", comm=comm) as f:
                f["y"].read_direct(y, cartslicer.slice_global_buffer_to_tile)
            if device == "gpu":
                y = xp.asarray(y)
        case "serial":
            with h5py.File(obs_path, "r") as f:
                y = xp.asarray(f["y"])
            state_shape = gt_shape
        case _:
            raise ValueError(f"Unknown run mode: {mode}")

    model_params = GaussianInpaintingTvParameters(y, mask, sigma2, reg_coef, split_coef)

    match device:
        case "cpu":
            X = PSGLA(state_shape, step_size_X, dtype=y.dtype)
            Z = PSGLA((2, *state_shape), step_size_Z, dtype=y.dtype)
        case "gpu":
            X = GpuPSGLA(state_shape, step_size_X, dtype=y.dtype)
            Z = GpuPSGLA((2, *state_shape), step_size_Z, dtype=y.dtype)
        case _:
            raise ValueError(f"Unknown device: {device}")

    match mode:
        case "mpi":
            model = DistributedGaussianInpaintingTvModel(
                comm,
                np.asarray(gt_shape),
                grid_size,
                model_params,
                X,
                Z,
            )
            match device:
                case "cpu":
                    sampler = DistributedSampler(comm, sampler_params, model, logger)
                case "gpu":
                    # TODO: revise / generalise default gpu assignment
                    sampler = MultiGpuSampler(
                        comm,
                        sampler_params,
                        model,
                        logger,
                        comm.Get_rank() % xp.cuda.runtime.getDeviceCount(),
                    )
                case _:
                    raise ValueError(f"Unknown device: {device}")
        case "serial":
            model = GaussianInpaintingTvModel(model_params, X, Z)
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
    denoiser_params: dict,
    mode: str = "serial",
    device: str = "cpu",
):
    mask, sigma2, gt_shape = load_from_h5(obs_path)
    mask = fit_mask_shape(mask, gt_shape)

    eps = (
        denoiser_params["denoising_level"] ** 2
        if denoiser_params["denoising_level"] is not None
        else sigma2
    )
    step_size_X, lambda_ = compute_step_sizes_gaussian_inpainting_pnp(
        sigma2,
        reg_coef,
        L=denoiser_params.get("L", None) or 1.0,
        eps=eps,
    )

    match mode:
        case "mpi":
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()
            # MPI.Compute_dims(size, 2)
            grid_size = np.asarray([1] * (len(gt_shape) - 2) + [size, 1])
            mpi_cart_comm = comm.Create_cart(grid_size)
            ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))

            cartslicer = CartesianCommSlicer(
                ranknd,
                grid_size,
                gt_shape,
                np.zeros(len(grid_size), dtype=int),
                np.zeros(len(grid_size), dtype=int),
            )
            state_shape = tuple(cartslicer.tile_size)

            mask = mask[cartslicer.slice_global_buffer_to_tile]

            dtype = read_dtype(obs_path, "y")
            y = np.empty(state_shape, dtype=dtype)
            interpolation = np.empty_like(y)
            with h5py.File(obs_path, "r", driver="mpio", comm=comm) as f:
                f["y"].read_direct(y, cartslicer.slice_global_buffer_to_tile)
                f["interpolation"].read_direct(
                    interpolation, cartslicer.slice_global_buffer_to_tile
                )
            if device == "gpu":
                y = xp.asarray(y)
                interpolation = xp.asarray(interpolation)
        case "serial":
            with h5py.File(obs_path, "r") as f:
                y = xp.asarray(f["y"])
                interpolation = xp.asarray(f["interpolation"])
            state_shape = gt_shape
        case _:
            raise ValueError(f"Unknown run mode: {mode}")

    model_params = GaussianInpaintingPnpParameters(y, mask, sigma2, reg_coef)

    match device:
        case "cpu":
            raise NotImplementedError("PnP is not implemented for CPU device.")
        case "gpu":
            X = GpuPnpULA(
                state_shape,
                step_size_X,
                reg_coef,
                sigma2,
                lambda_,
                dtype=y.dtype,
                initialization=interpolation,
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
            model = DistributedGaussianInpaintingPnpModel(
                comm,
                np.asarray(gt_shape),
                model_params,
                X,
                denoiser,
            )
            match device:
                case "cpu":
                    sampler = DistributedSampler(comm, sampler_params, model, logger)
                case "gpu":
                    # TODO: revise / generalise default gpu assignment
                    sampler = MultiGpuSampler(
                        comm,
                        sampler_params,
                        model,
                        logger,
                        comm.Get_rank() % xp.cuda.runtime.getDeviceCount(),
                    )
                case _:
                    raise ValueError(f"Unknown device: {device}")
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
            model = GaussianInpaintingPnpModel(model_params, X, denoiser)
            Sampler = SerialSampler if device == "cpu" else GpuSampler
            sampler = Sampler(sampler_params, model, logger)
        case _:
            raise ValueError(f"Unknown run mode: {mode}")

    sampler.sample()
