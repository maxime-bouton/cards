import logging
from pathlib import Path

import h5py
import numpy as np

from mcmc.backend import xp
from mcmc.models.gaussian_deconvolution_pnp_model import (
    DistributedGaussianDeconvolutionPnpModel,
    GaussianDeconvolutionPnpModel,
    GaussianDeconvolutionPnpParams,
)
from mcmc.models.gaussian_deconvolution_tv_model import (
    DistributedGaussianDeconvolutionTvModel,
    GaussianDeconvolutionTvModel,
    GaussianDeconvolutionTvParams,
)
from mcmc.operators.mpi_dft_convolution import MpiDftConvolution
from mcmc.sampler.base_sampler import SamplerParameters
from mcmc.sampler.distributed_sampler import DistributedSampler
from mcmc.sampler.gpu_sampler import GpuSampler
from mcmc.sampler.multi_gpu_sampler import MultiGpuSampler
from mcmc.sampler.serial_sampler import SerialSampler
from mcmc.transition_kernel.gpu_pnp_ula import GpuPnpULA
from mcmc.transition_kernel.gpu_psgla import GpuPSGLA
from mcmc.transition_kernel.psgla import PSGLA
from mcmc.utils.utils_observations import fit_kernel_shape


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

    x = 0.99 / (1.0 / split_coef + xp.max(xp.abs(xp.fft.rfft2(kernel))) ** 2 / sigma2)
    z = 0.99 * split_coef
    return x, z


def compute_step_sizes_gaussian_deconvolution_pnp(
    sigma2: float,
    kernel: xp.ndarray,
    reg_coef: float,
    L: float,
    eps: float,
) -> tuple[float, float]:
    Ly = xp.max(xp.abs(xp.fft.rfft2(kernel))) ** 2 / sigma2
    lambda_ = 0.99 / (2 * reg_coef * L / eps + 4 * Ly)
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
        sigma2 = data_file["sigma2"][()]  # type: ignore
        gt_shape = data_file["x"].shape  # type: ignore
        obs_shape = data_file["y"].shape  # type: ignore
    return kernel, sigma2, gt_shape, obs_shape  # type: ignore


def compute_tv(
    logger: logging.Logger,
    sampler_params: SamplerParameters,
    obs_path: str,
    reg_coef: float,
    split_coef: float,
    mode: str = "serial-cpu",
):
    kernel, sigma2, gt_shape, _ = load_from_h5(obs_path)
    step_size_X, step_size_Z = compute_step_sizes_gaussian_deconvolution_tv(
        split_coef, sigma2, kernel
    )

    kernel = fit_kernel_shape(kernel, gt_shape)

    if "mpi" in mode:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        # MPI.Compute_dims(size, 2)
        grid_size = np.asarray([1] * (len(gt_shape) - 2) + [size, 1])

        op = MpiDftConvolution(np.asarray(gt_shape), kernel, comm, grid_size)
        y = np.empty(op.adjoint_communicator.cartslicer.tile_size, dtype=kernel.dtype)
        with h5py.File(obs_path, "r", driver="mpio", comm=comm) as f:
            f["y"].read_direct(  # type: ignore
                y,
                op.adjoint_communicator.cartslicer.slice_global_buffer_to_tile,
            )
        if "gpu" in mode:
            y = xp.asarray(y)

        state_shape = tuple(op.direct_communicator.cartslicer.tile_size)

    else:
        with h5py.File(obs_path, "r") as f:
            y = xp.asarray(f["y"])
        state_shape = gt_shape

    model_params = GaussianDeconvolutionTvParams(
        y,
        kernel,
        sigma2,
        reg_coef,
        split_coef,
    )

    if "cpu" in mode:
        X = PSGLA(state_shape, step_size_X, dtype=y.dtype)
        Z = PSGLA((2, *state_shape), step_size_Z, dtype=y.dtype)
    elif "gpu" in mode:
        X = GpuPSGLA(state_shape, step_size_X, dtype=y.dtype)
        Z = GpuPSGLA((2, *state_shape), step_size_Z, dtype=y.dtype)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if "mpi" in mode:
        model = DistributedGaussianDeconvolutionTvModel(
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
        model = GaussianDeconvolutionTvModel(model_params, X, Z)
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
    kernel, sigma2, gt_shape, _ = load_from_h5(obs_path)

    eps = (
        denoiser_params["denoising_level"] ** 2
        if denoiser_params["denoising_level"] is not None
        else sigma2
    )
    step_size_X, lambda_ = compute_step_sizes_gaussian_deconvolution_pnp(
        sigma2,
        kernel,
        reg_coef,
        L=denoiser_params.get("L", None) or 1.0,
        eps=eps,
    )

    kernel = fit_kernel_shape(kernel, gt_shape)

    if "mpi" in mode:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        grid_size = np.asarray([1] * (len(gt_shape) - 2) + [size, 1])
        tile_range = None

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
                tile_range = (
                    denoiser.tail_conv.adjoint_communicator.cartslicer.tile_range
                )
            case _:
                raise ValueError(f"Unknown denoiser type: {denoiser_params['type']}")

        op = MpiDftConvolution(
            np.asarray(gt_shape),
            kernel,
            comm,
            grid_size,
            tile_range=tile_range,
        )
        y = np.empty(op.adjoint_communicator.cartslicer.tile_size, dtype=kernel.dtype)
        with h5py.File(obs_path, "r", driver="mpio", comm=comm) as f:
            f["y"].read_direct(  # type: ignore
                y,
                op.adjoint_communicator.cartslicer.slice_global_buffer_to_tile,
            )
        if "gpu" in mode:
            y = xp.asarray(y)

        state_shape = tuple(op.direct_communicator.cartslicer.tile_size)
        del op
    else:
        with h5py.File(obs_path, "r") as f:
            y = xp.asarray(f["y"])
        state_shape = gt_shape

    model_params = GaussianDeconvolutionPnpParams(y, kernel, sigma2, reg_coef)

    if "cpu" in mode:
        raise NotImplementedError("PNP is not implemented for CPU mode.")
    elif "gpu" in mode:
        X = GpuPnpULA(
            state_shape, step_size_X, reg_coef, sigma2, lambda_, dtype=y.dtype
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if "mpi" in mode:
        model = DistributedGaussianDeconvolutionPnpModel(
            comm,
            np.asarray(gt_shape),
            grid_size,
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
        model = GaussianDeconvolutionPnpModel(model_params, X, denoiser)
        Sampler = SerialSampler if "cpu" in mode else GpuSampler
        sampler = Sampler(sampler_params, model, logger)

    sampler.sample()
