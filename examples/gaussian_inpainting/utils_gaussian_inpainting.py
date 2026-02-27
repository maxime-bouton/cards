import logging

import h5py
import numpy as np

from mcmc.backend import xp
from mcmc.models.gaussian_inpainting_pnp_model import (
    DistributedGaussianInpaintingPnpModel,
    GaussianInpaintingPnpModel,
    GaussianInpaintingPnpParameters,
)
from mcmc.models.gaussian_inpainting_tv_model import (
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
from mcmc.transition_kernel.gpu_pnp_ula import GpuPnpULA
from mcmc.transition_kernel.gpu_psgla import GpuPSGLA
from mcmc.transition_kernel.psgla import PSGLA
from mcmc.utils.utils_img import read_dtype
from mcmc.utils.utils_observations import fit_mask_shape


def compute_step_sizes_gaussian_inpainting_tv(
    split_coef: float,
    sigma2: float,
) -> tuple[float, float]:
    x = 0.99 / (1.0 / split_coef + 1.0 / sigma2)
    z = 0.99 * split_coef
    return x, z


def compute_step_sizes_gaussian_inpainting_pnp(
    sigma2: float,
    reg_coef: float,
    L: float,
    eps: float,
) -> tuple[float, float]:
    Ly = 1 / sigma2
    lambda_ = 0.99 / (2 * reg_coef * L / eps + 4 * Ly)
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
    mode: str = "serial-cpu",
):
    # TODO: mask is loaded entirely on all processes in MPI mode
    mask, sigma2, gt_shape = load_from_h5(obs_path)
    step_size_X, step_size_Z = compute_step_sizes_gaussian_inpainting_tv(
        split_coef, sigma2
    )

    mask = fit_mask_shape(mask, gt_shape)

    if "mpi" in mode:
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
        if "gpu" in mode:
            y = xp.asarray(y)
            interpolation = xp.asarray(interpolation)
    else:
        with h5py.File(obs_path, "r") as f:
            y = xp.asarray(f["y"])
            interpolation = xp.asarray(f["interpolation"])
        state_shape = gt_shape

    model_params = GaussianInpaintingTvParameters(y, mask, sigma2, reg_coef, split_coef)

    if "cpu" in mode:
        X = PSGLA(state_shape, step_size_X, dtype=y.dtype)
        Z = PSGLA((2, *state_shape), step_size_Z, dtype=y.dtype)
    elif "gpu" in mode:
        X = GpuPSGLA(
            state_shape, step_size_X, dtype=y.dtype, initialization=interpolation
        )
        Z = GpuPSGLA(
            (2, *state_shape), step_size_Z, dtype=y.dtype, initialization=interpolation
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if "mpi" in mode:
        model = DistributedGaussianInpaintingTvModel(
            comm,
            np.asarray(gt_shape),
            grid_size,
            model_params,
            X,
            Z,
        )
        MpiSampler = DistributedSampler if "cpu" in mode else MultiGpuSampler
        sampler = MpiSampler(comm, sampler_params, model, logger)
    else:
        model = GaussianInpaintingTvModel(model_params, X, Z)
        Sampler = SerialSampler if "cpu" in mode else GpuSampler
        sampler = Sampler(sampler_params, model, logger)

    sampler.sample()


def compute_pnp(
    logger: logging.Logger,
    sampler_params: SamplerParameters,
    obs_path: str,
    reg_coef: float,
    denoiser_params: dict,
    mode: str = "serial-cpu",
):
    mask, sigma2, gt_shape = load_from_h5(obs_path)

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

    mask = fit_mask_shape(mask, gt_shape)

    if "mpi" in mode:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        # MPI.Compute_dims(size, 2)
        grid_size = np.asarray([1] * (len(gt_shape) - 2) + [size, 1])
        mpi_cart_comm = comm.Create_cart(grid_size)
        ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))

        match denoiser_params["type"]:
            case "ddfb":
                from mcmc.denoisers.mpi_ddfb import MpiDDFB

                denoiser = MpiDDFB(
                    comm,
                    grid_size,
                    image_size=np.asarray(gt_shape),
                    n_layers=denoiser_params["n_layers"],
                    n_features=denoiser_params["n_features"],
                )
            case "dncnn":
                from mcmc.denoisers.mpi_dncnn import MpiDnCNN

                denoiser = MpiDnCNN(comm, grid_size, image_size=np.asarray(gt_shape))
            case "drunet":
                from mcmc.denoisers.mpi_drunet import MpiDRUNet

                denoiser = MpiDRUNet(comm, grid_size, image_size=np.asarray(gt_shape))
            case _:
                raise ValueError(f"Unknown denoiser type: {denoiser_params['type']}")

        state_shape = denoiser.state_shape
        mask = mask[denoiser.global_to_tile_slice]

        dtype = read_dtype(obs_path, "y")

        y = np.empty(state_shape, dtype=dtype)
        interpolation = np.empty_like(y)
        with h5py.File(obs_path, "r", driver="mpio", comm=comm) as f:
            f["y"].read_direct(y, denoiser.global_to_tile_slice)
            f["interpolation"].read_direct(interpolation, denoiser.global_to_tile_slice)
        if "gpu" in mode:
            y = xp.asarray(y)
            interpolation = xp.asarray(interpolation)
    else:
        with h5py.File(obs_path, "r") as f:
            y = xp.asarray(f["y"])
            interpolation = xp.asarray(f["interpolation"])
        state_shape = gt_shape

    model_params = GaussianInpaintingPnpParameters(y, mask, sigma2, reg_coef)

    if "cpu" in mode:
        raise NotImplementedError("PNP is not implemented for CPU mode.")
    elif "gpu" in mode:
        X = GpuPnpULA(
            state_shape,
            step_size_X,
            reg_coef,
            sigma2,
            lambda_,
            dtype=y.dtype,
            initialization=interpolation,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if "mpi" in mode:
        model = DistributedGaussianInpaintingPnpModel(
            comm,
            np.asarray(gt_shape),
            model_params,
            X,
            denoiser,
        )
        MpiSampler = DistributedSampler if "cpu" in mode else MultiGpuSampler
        sampler = MpiSampler(comm, sampler_params, model, logger)
    else:
        match denoiser_params["type"]:
            case "ddfb":
                from mcmc.denoisers.serial_ddfb import SerialDDFB

                denoiser = SerialDDFB(
                    image_size=np.asarray(gt_shape),
                    n_layers=denoiser_params["n_layers"],
                    n_features=denoiser_params["n_features"],
                )
            case "dncnn":
                from mcmc.denoisers.serial_dncnn import SerialDnCNN

                denoiser = SerialDnCNN(image_size=np.asarray(gt_shape))
            case "drunet":
                from mcmc.denoisers.serial_drunet import SerialDRUNet

                denoiser = SerialDRUNet(image_size=np.asarray(gt_shape))
            case _:
                raise ValueError(f"Unknown denoiser type: {denoiser_params['type']}")
        model = GaussianInpaintingPnpModel(model_params, X, denoiser)
        Sampler = SerialSampler if "cpu" in mode else GpuSampler
        sampler = Sampler(sampler_params, model, logger)

    sampler.sample()
