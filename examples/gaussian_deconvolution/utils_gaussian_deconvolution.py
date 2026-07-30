r"""Utility functions to set the Gaussian deconvolution example script for the experiments reported in :cite:p:`Bouton2026` (synthetic data generation, sampling and post-processing steps)."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.

# FIXME: use DataManager instead to hide details on data loading

import logging
from pathlib import Path

import h5py
import numpy as np

import cards.backend as xp
from cards.estimators.base_estimator_builder import BaseEstimatorBuilder
from cards.estimators.mmse_var_builder import MMSEVarBuilder
from cards.models.gaussian_deconvolution_pnp_model import (
    DistributedGaussianDeconvolutionPnpModel,
    GaussianDeconvolutionPnpModel,
    GaussianDeconvolutionPnpParams,
)
from cards.models.gaussian_deconvolution_tv_model import (
    DistributedGaussianDeconvolutionTvModel,
    GaussianDeconvolutionTvModel,
    GaussianDeconvolutionTvParams,
)
from cards.operators.dft_convolution import DftConvolution
from cards.operators.mpi_dft_convolution import MpiDftConvolution
from cards.samplers import (
    DistributedCpuSampler,
    DistributedGpuSampler,
    SamplerParameters,
    SerialCpuSampler,
    SerialGpuSampler,
)
from cards.transition_kernels.gpu_pnp_ula import GpuPnpULA
from cards.transition_kernels.gpu_psgla import GpuPSGLA
from cards.transition_kernels.psgla import PSGLA
from cards.utils.path_builder import (
    deconvolution_str,
    gaussian_str,
    obs_dir,
)
from cards.utils.utils import extract_subset_from_dict
from cards.utils.utils_img import read_dtype, read_img_shape
from cards.utils.utils_observations import (
    apply_target_gaussian_noise,
    fit_kernel_shape,
    generate_gaussian_kernel,
    generate_motion_kernel,
    generate_observations,
    slice_linear_conv_to_original,
)


def gaussian_deconvolution_params(params: dict) -> dict:
    if "denoiser_params" in params:
        return extract_subset_from_dict(params, ["reg_coef", "denoiser_params"])
    else:
        return extract_subset_from_dict(params, ["reg_coef", "split_coef"])


def define_slices(params: dict) -> dict:
    img_size = read_img_shape(params["original_img_path"])
    # TODO: find a more elegant way to deal with image of different dimensions
    kernel_size = [1] * (len(img_size) - 2) + [params["kernel"]["size"]] * 2
    return {"slices": slice_linear_conv_to_original(img_size, kernel_size)}


# FIXME: revise to allow generation on GPU
def generate_gaussian_deconvolution_observations(
    original_img_path: str,
    kernel_params: dict,
    isnr: float,
    data_seed: int,
    obs_path: str,
    maximum: float = 1.0,
    mode: str = "serial",
    # device: str = "cpu",
):
    # data type and shapes
    gt_shape = read_img_shape(original_img_path)
    gt_size = np.asarray(gt_shape)
    dtype = read_dtype(original_img_path)

    data_size = gt_size.copy()
    # convolution affects only the last two dimensions (i.e., spatial dimensions)
    data_size[-2:] += np.asarray(kernel_params["size"], dtype=int) - 1

    # generate convolution kernel
    if kernel_params["type"] == "motion":
        kernel = generate_motion_kernel(
            kernel_params["size"],
            kernel_params["intensity"],
            dtype=dtype,
            rng=np.random.default_rng(data_seed),
        )
    else:
        kernel = generate_gaussian_kernel(
            kernel_params["size"],
            kernel_params["std"],
            dtype=dtype,
        )
    kernel = fit_kernel_shape(kernel, gt_size)

    # define numpy rng and operators
    match mode:
        case "mpi":
            from mpi4py import MPI

            from cards.operators.mpi_dft_convolution import MpiDftConvolution

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

            convolution_operator = MpiDftConvolution(gt_size, kernel, comm, grid_size)

        case "serial":
            from cards.operators.dft_convolution import DftConvolution

            seed = data_seed

            convolution_operator = DftConvolution(gt_size, kernel, data_size)

        case _:
            raise ValueError(f"Unknown run mode: {mode}")

    rng = np.random.default_rng(seed)

    # load ground truth image
    # FIXME: use DataManager instead to hide details?
    match mode:
        case "mpi":
            # file from which the ground-truth image is loaded
            with h5py.File(original_img_path, "r+", driver="mpio", comm=comm) as f:
                dset = f["x"]
                x = np.zeros(
                    convolution_operator.direct_communicator.cartslicer.tile_size,
                    dtype=dtype,
                )
                dset.read_direct(
                    x,
                    convolution_operator.direct_communicator.cartslicer.slice_global_buffer_to_tile,
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

    observations, normalized_img, extra_params = generate_observations(
        x,
        convolution_operator,
        apply_target_gaussian_noise,
        rng,
        maximum,
        isnr=isnr,
    )

    # save data
    # FIXME: use DataManager instead to hide details
    params_saved = {"kernel": kernel}
    params_saved.update({"isnr": isnr})
    params_saved.update(*extra_params)

    match mode:
        case "mpi":
            with h5py.File(obs_path, "w", driver="mpio", comm=comm) as file:
                dset_x = file.create_dataset("x", gt_size, dtype=normalized_img.dtype)
                dset_x[
                    convolution_operator.direct_communicator.cartslicer.slice_global_buffer_to_tile
                ] = (
                    normalized_img
                    if isinstance(normalized_img, np.ndarray)
                    or np.isscalar(normalized_img)
                    else normalized_img.get()
                )

                dset_y = file.create_dataset("y", data_size, dtype=observations.dtype)
                dset_y[
                    convolution_operator.adjoint_communicator.cartslicer.slice_global_buffer_to_tile
                ] = (
                    observations
                    if isinstance(observations, np.ndarray) or np.isscalar(observations)
                    else observations.get()
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
    application_str = deconvolution_str(params)
    obs_f = f"gaussian-deconvolution/{obs_dir(params, application_str, noise_str)}"
    model_params_f = application_params_dir(params)

    return obs_f, model_params_f


def compute_step_sizes_gaussian_deconvolution_tv(
    split_coef: float,
    sigma2: float,
    kernel: xp.ndarray,
) -> tuple[float, float]:
    """Compute the step sizes for the PSGLA transition kernel with TV prior.

    Parameters
    ----------
    split_coef : float
        Splitting coefficient, parameter of the model.
    sigma2 : float
        Variance of the noise, parameter of the model.
    kernel : xp.ndarray
        Kernel used in the deconvolution model.

    Returns
    -------
    tuple[float, float]
        Step sizes for the X and Z variables in the PSGLA transition kernel.
    """

    step_size_X = 0.99 / (
        8.0 / split_coef + xp.max(xp.abs(xp.fft.rfft2(kernel))) ** 2 / sigma2
    )
    step_size_Z = 0.99 * split_coef
    return step_size_X, step_size_Z


def compute_step_sizes_gaussian_deconvolution_pnp(
    sigma2: float,
    kernel: xp.ndarray,
    reg_coef: float,
    L: float,
    eps: float,
) -> tuple[float, float]:
    eps = eps or sigma2
    Ly = xp.max(xp.abs(xp.fft.rfft2(kernel))) ** 2 / sigma2
    lambda_ = 0.99 / (2 * L / eps + 4 * Ly)
    be = (reg_coef * L) / eps + 1 / lambda_ + Ly
    step_size_X = 0.99 / (3 * be)
    return step_size_X, lambda_


def load_from_h5(
    filename: str | Path,
) -> tuple[xp.ndarray, float, tuple[int, ...], tuple[int, ...]]:
    """Load the kernel, sigma2, and observation shape from an HDF5 file.

    Parameters
    ----------
    filename : str or Path
        Path to the HDF5 file containing the kernel, sigma2, and observations.

    Returns
    -------
    tuple[xp.ndarray, float, tuple[int, ...], tuple[int, ...]]
        A tuple containing the kernel as a cupy array, the sigma2 value as a float,
        the shape of the ground truth as a tuple of integers,
        and the shape of the observations as a tuple of integers.
    """
    with h5py.File(filename, "r") as data_file:
        kernel = xp.asarray(data_file["kernel"])
        sigma2 = data_file["sigma2"][()]
        gt_shape = data_file["x"].shape
        obs_shape = data_file["y"].shape
    return kernel, sigma2, gt_shape, obs_shape


def compute_tv(
    logger: logging.Logger,
    sampler_params: SamplerParameters,
    obs_path: str,
    reg_coef: float,
    split_coef: float,
    mode: str = "serial",
    device: str = "cpu",
):
    kernel, sigma2, gt_shape, _ = load_from_h5(obs_path)
    step_size_X, step_size_Z = compute_step_sizes_gaussian_deconvolution_tv(
        split_coef, sigma2, kernel
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
                f["y"].read_direct(
                    y,
                    op.adjoint_communicator.cartslicer.slice_global_buffer_to_tile,
                )
            if device == "gpu":
                y = xp.asarray(y)
            state_shape = tuple(op.direct_communicator.cartslicer.tile_size)
        case "serial":
            with h5py.File(obs_path, "r") as f:
                y = xp.asarray(f["y"])
            state_shape = gt_shape
            data_size = np.asarray(gt_shape) + np.asarray(kernel.shape, dtype=int) - 1
            op = DftConvolution(np.asarray(gt_shape), kernel, data_size)
        case _:
            raise ValueError(f"Unknown run mode: {mode}")

    model_params = GaussianDeconvolutionTvParams(
        y,
        kernel,
        sigma2,
        reg_coef,
        split_coef,
    )

    match device:
        case "cpu":
            X = PSGLA(state_shape, step_size_X, dtype=y.dtype)
            Z = PSGLA((2, *state_shape), step_size_Z, dtype=y.dtype)
        case "gpu":
            X = GpuPSGLA(state_shape, step_size_X, dtype=y.dtype)
            Z = GpuPSGLA((2, *state_shape), step_size_Z, dtype=y.dtype)
        case _:
            raise ValueError(f"Unknown device: {device}")

    estimators: list[BaseEstimatorBuilder] = [MMSEVarBuilder(X)]

    match mode:
        case "mpi":
            model = DistributedGaussianDeconvolutionTvModel(
                estimators,
                model_params,
                op,
                X,
                Z,
            )
            Sampler = (
                DistributedCpuSampler if device == "cpu" else DistributedGpuSampler
            )
            sampler = Sampler(comm, sampler_params, model, logger)
        case "serial":
            model = GaussianDeconvolutionTvModel(estimators, model_params, op, X, Z)
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
    kernel, sigma2, gt_shape, _ = load_from_h5(obs_path)

    eps = (
        denoiser_params["denoising_level"] ** 2
        if denoiser_params["denoising_level"] is not None
        else sigma2
    )
    L = denoiser_params.get("L", None) or 1.0
    step_size_X, lambda_ = compute_step_sizes_gaussian_deconvolution_pnp(
        sigma2,
        kernel,
        reg_coef,
        L,
        eps,
    )

    kernel = fit_kernel_shape(kernel, gt_shape)

    match mode:
        case "mpi":
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            grid_size = np.asarray([1] * (len(gt_shape) - 2) + [size, 1])
            tile_range = None

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
                    tile_range = (
                        denoiser.tail_conv.adjoint_communicator.cartslicer.tile_range
                    )

                case _:
                    raise ValueError(
                        f"Unknown denoiser type: {denoiser_params['type']}"
                    )

            op = MpiDftConvolution(
                np.asarray(gt_shape), kernel, comm, grid_size, tile_range=tile_range
            )
            y = np.empty(
                op.adjoint_communicator.cartslicer.tile_size, dtype=kernel.dtype
            )
            with h5py.File(obs_path, "r", driver="mpio", comm=comm) as f:
                f["y"].read_direct(
                    y,
                    op.adjoint_communicator.cartslicer.slice_global_buffer_to_tile,
                )
            if device == "gpu":
                y = xp.asarray(y)
            state_shape = tuple(op.direct_communicator.cartslicer.tile_size)
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

            with h5py.File(obs_path, "r") as f:
                y = xp.asarray(f["y"])
            state_shape = gt_shape
            data_size = np.asarray(gt_shape) + np.asarray(kernel.shape, dtype=int) - 1
            op = DftConvolution(np.asarray(gt_shape), kernel, data_size)
        case _:
            raise ValueError(f"Unknown run mode: {mode}")

    model_params = GaussianDeconvolutionPnpParams(y, kernel, sigma2, reg_coef)

    match device:
        case "cpu":
            raise NotImplementedError("PNP is not implemented for CPU mode.")
        case "gpu":
            X = GpuPnpULA(
                state_shape, step_size_X, reg_coef, sigma2, lambda_, dtype=y.dtype
            )
        case _:
            raise ValueError(f"Unknown device: {device}")

    estimators: list[BaseEstimatorBuilder] = [MMSEVarBuilder(X)]

    match mode:
        case "mpi":
            model = DistributedGaussianDeconvolutionPnpModel(
                estimators,
                model_params,
                op,
                X,
                denoiser,
            )

            Sampler = (
                DistributedCpuSampler if device == "cpu" else DistributedGpuSampler
            )
            sampler = Sampler(comm, sampler_params, model, logger)
        case "serial":
            model = GaussianDeconvolutionPnpModel(
                estimators,
                model_params,
                op,
                X,
                denoiser,
            )
            Sampler = SerialCpuSampler if device == "cpu" else SerialGpuSampler
            sampler = Sampler(sampler_params, model, logger)
        case _:
            raise ValueError(f"Unknown run mode: {mode}")

    sampler.sample()
