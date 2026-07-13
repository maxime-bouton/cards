r"""Utility functions to set the Gaussian inpainting example script for the experiments reported in :cite:p:`Bouton2025` (synthetic data generation, sampling and post-processing steps)."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.

# FIXME: use DataManager instead to hide details on data loading

import logging

import h5py
import numpy as np
from scipy import interpolate

import cards.backend as xp
from cards.estimators.base_estimator_builder import BaseEstimatorBuilder
from cards.estimators.ci_builder import CIBuilder
from cards.estimators.mmse_var_builder import MMSEVarBuilder
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
from cards.samplers.base_sampler import SamplerParameters
from cards.samplers.distributed_cpu_sampler import DistributedCpuSampler
from cards.samplers.distributed_gpu_sampler import DistributedGpuSampler
from cards.samplers.serial_cpu_sampler import SerialCpuSampler
from cards.samplers.serial_gpu_sampler import SerialGpuSampler
from cards.slicers.cartesian_comm_slicer import CartesianCommSlicer
from cards.transition_kernels.gpu_pnp_ula import GpuPnpULA
from cards.transition_kernels.gpu_psgla import GpuPSGLA
from cards.transition_kernels.psgla import PSGLA
from cards.utils.path_builder import gaussian_str, inpainting_str, obs_dir
from cards.utils.utils import extract_subset_from_dict
from cards.utils.utils_img import read_dtype, read_img_shape  # load_img
from cards.utils.utils_observations import (
    apply_target_gaussian_noise,
    fit_mask_shape,
    generate_observations,
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
    r"""Interpolate masked values in an image using cubic spline interpolation.
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
        if xp.get_backend() == "cupy":
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


# FIXME: revise to allow generation on GPU
def generate_inpainting_observations(
    original_img_path: str,
    mask_loss: float,
    isnr: float,
    data_seed: int,
    obs_path: str,
    maximum: float = 1.0,
    mode: str = "serial",
    # device: str = "cpu",
):
    # data type and shapes
    gt_shape = read_img_shape(original_img_path)
    gt_size = xp.asarray(gt_shape, dtype=int)
    dtype = read_dtype(original_img_path)

    # define numpy rng and cartesian slicer
    match mode:
        case "mpi":
            from mpi4py import MPI

            from cards.communicators.mpi_utils import get_ranknd
            from cards.slicers.cartesian_comm_slicer import CartesianCommSlicer

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            comm_size = comm.Get_size()

            if rank == 0:
                ss = np.random.SeedSequence(data_seed)
                # spawn off nworkers child SeedSequences to pass to child processes.
                child_seed = ss.spawn(comm_size)
            else:
                child_seed = None
            seed = comm.scatter(child_seed, root=0)

            # MPI.Compute_dims(size, 2)
            grid_size = np.asarray([1] * (len(gt_shape) - 2) + [comm_size, 1])

            # slicer to handle image and mask portions
            ranknd = get_ranknd(rank, grid_size)

            cartslicer = CartesianCommSlicer(
                ranknd,
                grid_size,
                gt_size,
                xp.zeros(len(grid_size), dtype=int),
                xp.zeros(len(grid_size), dtype=int),
            )
            local_size = cartslicer.tile_size

        case "serial":
            seed = data_seed
            local_size = gt_size

        case _:
            raise ValueError(f"Unknown run mode: {mode}")
    rng = np.random.default_rng(seed)

    # generate local mask and operator
    sz = [local_size[k].item() for k in range(local_size.size - 2, local_size.size)]
    mask = rng.random(sz) < (1 - mask_loss)
    mask_extended = fit_mask_shape(xp.asarray(mask), local_size)

    # load ground truth image
    # FIXME: use DataManager instead to hide details?
    match mode:
        case "mpi":
            # file from which the ground-truth image is loaded
            with h5py.File(original_img_path, "r+", driver="mpio", comm=comm) as f:
                dset = f["x"]
                x = np.zeros(
                    cartslicer.tile_size,
                    dtype=dtype,
                )
                dset.read_direct(
                    x,
                    cartslicer.slice_global_buffer_to_tile,
                    (np.s_[:], np.s_[:]),
                )
                x = xp.asarray(x)

        case "serial":
            with h5py.File(
                original_img_path,
                "r+",
            ) as f:
                x = xp.asarray(f["x"][:])

        case _:
            raise ValueError(f"Unknown run mode: {mode}")

    inpainting_operator = Masking(mask_extended)
    observations, normalized_img, extra_params = generate_observations(
        x,
        inpainting_operator,
        apply_target_gaussian_noise,
        rng,
        maximum,
        isnr=isnr,
    )

    # NOTE: in distributed setting, only local interpolation (not equivalent to serial interpolation)
    interpolated_observations = interpolate_masked_image_cubic(
        observations, mask_extended
    ).clip(0, 1)

    # save data
    params_saved = {"mask": mask}
    params_saved.update({"isnr": isnr})
    params_saved.update(*extra_params)

    match mode:
        case "mpi":
            # file from which the ground-truth image is loaded
            with h5py.File(obs_path, "w", driver="mpio", comm=comm) as file:
                dset_x = file.create_dataset("x", gt_size, dtype=normalized_img.dtype)
                dset_x[cartslicer.slice_global_buffer_to_tile] = (
                    normalized_img
                    if isinstance(normalized_img, np.ndarray)
                    or np.isscalar(normalized_img)
                    else normalized_img.get()
                )

                dset_y = file.create_dataset("y", gt_size, dtype=observations.dtype)
                dset_y[cartslicer.slice_global_buffer_to_tile] = (
                    observations
                    if isinstance(observations, np.ndarray) or np.isscalar(observations)
                    else observations.get()
                )

                dset_interpolated_y = file.create_dataset(
                    "interpolation", gt_size, dtype=interpolated_observations.dtype
                )
                dset_interpolated_y[cartslicer.slice_global_buffer_to_tile] = (
                    interpolated_observations
                    if isinstance(interpolated_observations, np.ndarray)
                    or np.isscalar(interpolated_observations)
                    else interpolated_observations.get()
                )

            if rank == 0:
                with h5py.File(obs_path, "r+") as file:
                    file["seed_data"] = data_seed

                    for key, value in params_saved.items():
                        file[key] = (
                            value
                            if isinstance(value, np.ndarray) or np.isscalar(value)
                            else value.get()
                        )

        case "serial":
            with h5py.File(obs_path, "w") as file:
                file["x"] = (
                    normalized_img
                    if isinstance(normalized_img, np.ndarray)
                    or np.isscalar(normalized_img)
                    else normalized_img.get()
                )
                file["y"] = (
                    observations
                    if isinstance(observations, np.ndarray) or np.isscalar(observations)
                    else observations.get()
                )
                file["interpolation"] = (
                    interpolated_observations
                    if isinstance(interpolated_observations, np.ndarray)
                    or np.isscalar(interpolated_observations)
                    else interpolated_observations.get()
                )
                file["seed_data"] = data_seed

                for key, value in params_saved.items():
                    file[key] = (
                        value
                        if isinstance(value, np.ndarray) or np.isscalar(value)
                        else value.get()
                    )
        case _:
            raise ValueError(f"Unknown run mode: {mode}")


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
    step_size_X = 0.99 * 1.0 / (8.0 / split_coef + 1.0 / sigma2)
    step_size_Z = 0.99 * split_coef
    return step_size_X, step_size_Z


def compute_step_sizes_gaussian_inpainting_pnp(
    sigma2: float,
    reg_coef: float,
    L: float,
    eps: float,
) -> tuple[float, float]:
    Ly = 1 / sigma2
    lambda_ = 0.99 / (2 * L / eps + 4 * Ly)
    be = (reg_coef * L) / eps + 1 / lambda_ + Ly
    step_size_X = 0.99 / (3 * be)
    return step_size_X, lambda_


def load_from_h5(filename) -> tuple[float, tuple[int, ...]]:
    with h5py.File(filename, "r") as data_file:
        sigma2 = data_file["sigma2"][()]
        gt_shape = data_file["x"].shape
    return sigma2, gt_shape


def compute_tv(
    logger: logging.Logger,
    sampler_params: SamplerParameters,
    obs_path: str,
    reg_coef: float,
    split_coef: float,
    mode: str = "serial",
    device: str = "cpu",
):
    sigma2, gt_shape = load_from_h5(obs_path)
    step_size_X, step_size_Z = compute_step_sizes_gaussian_inpainting_tv(
        split_coef, sigma2
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

            mask = np.empty(state_shape[-2:], dtype=int)

            dtype = read_dtype(obs_path, "y")
            y = np.empty(state_shape, dtype=dtype)
            interpolation = np.empty_like(y)

            with h5py.File(obs_path, "r", driver="mpio", comm=comm) as f:
                f["y"].read_direct(y, cartslicer.slice_global_buffer_to_tile)

                # NOTE: mask is only 2D, slices accommodate up to 3D
                f["mask"].read_direct(mask, cartslicer.slice_global_buffer_to_tile[-2:])

                f["interpolation"].read_direct(
                    interpolation, cartslicer.slice_global_buffer_to_tile
                )
            if device == "gpu":
                y = xp.asarray(y)
                mask = xp.asarray(mask)
                interpolation = xp.asarray(interpolation)
        case "serial":
            with h5py.File(obs_path, "r") as f:
                y = xp.asarray(f["y"])
                mask = xp.asarray(f["mask"])
                interpolation = xp.asarray(f["interpolation"])
            state_shape = gt_shape
        case _:
            raise ValueError(f"Unknown run mode: {mode}")

    mask = fit_mask_shape(mask, state_shape)
    model_params = GaussianInpaintingTvParameters(y, mask, sigma2, reg_coef, split_coef)

    match device:
        case "cpu":
            X = PSGLA(
                state_shape,
                step_size_X,
                dtype=y.dtype,
                initial_value=interpolation,
            )
            Z = PSGLA((2, *state_shape), step_size_Z, dtype=y.dtype)
        case "gpu":
            X = GpuPSGLA(
                state_shape, step_size_X, dtype=y.dtype, initial_value=interpolation
            )
            Z = GpuPSGLA(
                (2, *state_shape),
                step_size_Z,
                dtype=y.dtype,
            )
        case _:
            raise ValueError(f"Unknown device: {device}")

    estimators: list[BaseEstimatorBuilder] = [
        MMSEVarBuilder(X),
        CIBuilder(X, all_samples=True),
    ]

    match mode:
        case "mpi":
            model = DistributedGaussianInpaintingTvModel(
                estimators,
                model_params,
                X,
                Z,
                comm,
                grid_size,
                np.asarray(gt_shape),
            )
            Sampler = (
                DistributedCpuSampler if device == "cpu" else DistributedGpuSampler
            )
            sampler = Sampler(comm, sampler_params, model, logger)
        case "serial":
            model = GaussianInpaintingTvModel(estimators, model_params, X, Z)
            Sampler = SerialCpuSampler if device == "cpu" else SerialGpuSampler
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
    sigma2, gt_shape = load_from_h5(obs_path)

    eps = (
        denoiser_params["denoising_level"] ** 2
        if denoiser_params["denoising_level"] is not None
        else sigma2
    )
    L = denoiser_params.get("L", None) or 1.0
    step_size_X, lambda_ = compute_step_sizes_gaussian_inpainting_pnp(
        sigma2,
        reg_coef,
        L,
        eps,
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

            mask = np.empty(state_shape[-2:], dtype=int)

            dtype = read_dtype(obs_path, "y")
            y = np.empty(state_shape, dtype=dtype)
            interpolation = np.empty_like(y)

            with h5py.File(obs_path, "r", driver="mpio", comm=comm) as f:
                f["y"].read_direct(y, cartslicer.slice_global_buffer_to_tile)

                # NOTE: mask is only 2D, slices accommodate up to 3D
                f["mask"].read_direct(mask, cartslicer.slice_global_buffer_to_tile[-2:])

                f["interpolation"].read_direct(
                    interpolation, cartslicer.slice_global_buffer_to_tile
                )
            if device == "gpu":
                y = xp.asarray(y)
                mask = xp.asarray(mask)
                interpolation = xp.asarray(interpolation)
        case "serial":
            with h5py.File(obs_path, "r") as f:
                y = xp.asarray(f["y"])
                mask = xp.asarray(f["mask"])
                interpolation = xp.asarray(f["interpolation"])
            state_shape = gt_shape
        case _:
            raise ValueError(f"Unknown run mode: {mode}")

    mask = fit_mask_shape(mask, state_shape)
    model_params = GaussianInpaintingPnpParameters(y, mask, sigma2, reg_coef)

    match device:
        case "cpu":
            raise NotImplementedError("PnP not implemented for CPU devices yet.")
        case "gpu":
            X = GpuPnpULA(
                state_shape,
                step_size_X,
                reg_coef,
                sigma2,
                lambda_,
                dtype=y.dtype,
                initial_value=interpolation,
            )
        case _:
            raise ValueError(f"Unknown device: {device}")

    estimators: list[BaseEstimatorBuilder] = [
        MMSEVarBuilder(X),
        CIBuilder(X, all_samples=False),
    ]

    match mode:
        case "mpi":
            match denoiser_params["type"]:
                case "ddfb":
                    from cards.denoisers.distributed_ddfb import DistributedDDFB

                    denoiser = DistributedDDFB(
                        comm,
                        grid_size,
                        image_size=np.asarray(gt_shape),
                        n_layers=denoiser_params["n_layers"],
                        n_features=denoiser_params["n_features"],
                    )
                case "dncnn":
                    from cards.denoisers.distributed_dncnn import DistributedDnCNN

                    denoiser = DistributedDnCNN(
                        comm, grid_size, image_size=np.asarray(gt_shape)
                    )
                case "drunet":
                    from cards.denoisers.distributed_drunet import DistributedDRUNet

                    denoiser = DistributedDRUNet(
                        comm, grid_size, image_size=np.asarray(gt_shape)
                    )
                case _:
                    raise ValueError(
                        f"Unknown denoiser type: {denoiser_params['type']}"
                    )
            model = DistributedGaussianInpaintingPnpModel(
                estimators,
                model_params,
                X,
                denoiser,
                np.asarray(gt_shape),
            )
            Sampler = (
                DistributedCpuSampler if device == "cpu" else DistributedGpuSampler
            )
            sampler = Sampler(comm, sampler_params, model, logger)
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
            model = GaussianInpaintingPnpModel(estimators, model_params, X, denoiser)
            Sampler = SerialCpuSampler if device == "cpu" else SerialGpuSampler
            sampler = Sampler(sampler_params, model, logger)
        case _:
            raise ValueError(f"Unknown run mode: {mode}")

    sampler.sample()
