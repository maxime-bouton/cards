from dataclasses import dataclass

import numpy as np

import cards.backend as xp
from cards.core.execution_context import ExecutionContext
from cards.denoisers.base_denoiser import BaseDenoiser
from cards.denoisers.mpi_ddfb import MpiDDFB
from cards.denoisers.mpi_dncnn import MpiDnCNN
from cards.denoisers.mpi_drunet import MpiDRUNet
from cards.denoisers.serial_ddfb import SerialDDFB
from cards.denoisers.serial_dncnn import SerialDnCNN
from cards.denoisers.serial_drunet import SerialDRUNet
from cards.operators.dft_convolution import DftConvolution
from cards.operators.mpi_dft_convolution import MpiDftConvolution
from cards.utils.utils_img import read_dtype, read_img_shape
from cards.utils.utils_observations import (
    fit_kernel_shape,
    generate_gaussian_kernel,
    generate_motion_kernel,
    load_img,
)


@dataclass
class PnpDeconvGeometry:
    kernel: xp.ndarray
    gt_shape: tuple[int, ...]
    denoiser: BaseDenoiser
    operator: DftConvolution | MpiDftConvolution
    tile_size: tuple[int, ...]


def build_convolution_operator(
    gt_shape: tuple[int, ...],
    grid_size: np.ndarray,
    kernel: xp.ndarray,
    ctx: ExecutionContext,
    tile_range: np.ndarray | None = None,
) -> DftConvolution | MpiDftConvolution:
    if ctx.is_mpi:
        return MpiDftConvolution(
            np.asarray(gt_shape),
            kernel,
            ctx.comm,
            grid_size,
            tile_range=tile_range,
        )
    data_size = np.asarray(gt_shape) + np.asarray(kernel.shape, dtype=int) - 1
    return DftConvolution(np.asarray(gt_shape), kernel, data_size)


def build_denoiser(
    params: dict,
    grid_size: np.ndarray,
    img_shape: tuple[int, ...],
    ctx: ExecutionContext,
) -> tuple[BaseDenoiser, np.ndarray | None]:
    if not ctx.is_gpu:
        raise ValueError("CPU not supported for Gaussian Deconvolution models with PnP")
    img_size = np.asarray(img_shape)
    tile_range = None
    match (params["type"], ctx.is_mpi):
        case ("ddfb", False):
            denoiser = SerialDDFB(img_size, params["n_layers"], params["n_features"])
        case ("ddfb", True):
            denoiser = MpiDDFB(
                ctx.comm,
                grid_size,
                img_size,
                params["n_layers"],
                params["n_features"],
            )
        case ("dncnn", False):
            denoiser = SerialDnCNN(img_size)
        case ("dncnn", True):
            denoiser = MpiDnCNN(ctx.comm, grid_size, img_size)
        case ("drunet", False):
            denoiser = SerialDRUNet(img_size)
        case ("drunet", True):
            denoiser = MpiDRUNet(ctx.comm, grid_size, img_size)
            tile_range = denoiser.tile_range
        case _:
            raise ValueError(f"Unknown denoiser type '{params['type']}'.")
    return denoiser, tile_range


def build_kernel(obs_cfg: dict, dtype: np.dtype | None = None) -> xp.ndarray:
    kernel_cfg = obs_cfg["kernel"]
    if kernel_cfg.get("path", None):
        return load_img(kernel_cfg["path"], dtype=dtype)

    data_seed = obs_cfg["seed_data"]
    if kernel_cfg["type"] == "motion":
        kernel = generate_motion_kernel(
            kernel_cfg["size"],
            kernel_cfg["intensity"],
            dtype=dtype,
            rng=np.random.default_rng(data_seed),
        )
    else:
        kernel = generate_gaussian_kernel(
            kernel_cfg["size"],
            kernel_cfg["std"],
            dtype=dtype,
        )
    return kernel


class PnpDeconvGeometryHook:
    def build_geometry(self, cfg: dict, ctx: ExecutionContext) -> PnpDeconvGeometry:
        obs_cfg = cfg["observations"]
        gt_path = obs_cfg["img_path"]
        gt_shape = read_img_shape(gt_path)
        dtype = read_dtype(gt_path)

        grid_size = ctx.generate_grid_size(len(gt_shape))
        kernel_2d = build_kernel(obs_cfg, dtype)
        kernel = fit_kernel_shape(kernel_2d, gt_shape)
        denoiser, tile_range = build_denoiser(
            cfg["parameters"]["denoiser"],
            grid_size,
            gt_shape,
            ctx,
        )
        op = build_convolution_operator(
            gt_shape,
            grid_size,
            kernel,
            ctx,
            tile_range=tile_range,
        )
        tile_size = (
            tuple(op.direct_communicator.cartslicer.tile_size)  # type: ignore
            if ctx.is_mpi
            else gt_shape
        )
        return PnpDeconvGeometry(kernel, gt_shape, denoiser, op, tile_size)
