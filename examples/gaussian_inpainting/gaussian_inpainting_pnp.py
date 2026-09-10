# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import interpolate

import cards.backend as xp
from cards.communicators.mpi_utils import get_ranknd
from cards.core.execution_context import ExecutionContext
from cards.core.layout import Layout
from cards.core.variable import Variable
from cards.denoisers.base_denoiser import BaseDenoiser
from cards.denoisers.distributed_ddfb import DistributedDDFB
from cards.denoisers.distributed_dncnn import DistributedDnCNN
from cards.denoisers.distributed_drunet import DistributedDRUNet
from cards.denoisers.serial_ddfb import SerialDDFB
from cards.denoisers.serial_dncnn import SerialDnCNN
from cards.denoisers.serial_drunet import SerialDRUNet
from cards.estimators.base_estimator import BaseEstimator
from cards.estimators.ci import CI
from cards.estimators.mmse_var import MMSEVar
from cards.io.io_manager import IOManager
from cards.models import (
    BaseModel,
    DistributedGaussianInpaintingPnpModel,
    GaussianInpaintingPnpModel,
)
from cards.models.gaussian_inpainting_pnp_model import (
    GaussianInpaintingPnpParameters,
)
from cards.operators.distributed_masking import DistributedMasking
from cards.operators.masking import Masking
from cards.random import create_rng
from cards.slicers.cartesian_comm_slicer import CartesianCommSlicer
from cards.transition_kernels.pnp_ula import CpuPnpULA, GpuPnpULA
from cards.utils.utils_img import read_img_shape
from cards.utils.utils_observations import (
    compute_sigma2_from_isnr,
    fit_mask_shape,
)

# TODO: test and documentation


@dataclass
class PnpInpaintingGeometry:
    grid_shape: tuple[int, ...]
    layout_x: Layout
    layout_y: Layout
    H: Masking | DistributedMasking
    D: BaseDenoiser
    mask: xp.ndarray


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


def build_masking_operator(
    mask: xp.ndarray,
    full_shape: tuple[int, ...],
    grid_shape: tuple[int, ...],
    ctx: ExecutionContext,
    cartslicer: CartesianCommSlicer,
) -> Masking | DistributedMasking:
    if ctx.is_mpi:
        return DistributedMasking(
            grid_shape,
            full_shape,
            mask,
            cartslicer,
        )
    return Masking(mask)


def build_denoiser(
    params: dict,
    full_shape: tuple[int, ...],
    grid_shape: tuple[int, ...],
    ctx: ExecutionContext,
) -> tuple[BaseDenoiser, np.ndarray | None]:
    tile_range = None
    match (params["type"], ctx.is_mpi):
        case ("ddfb", False):
            denoiser = SerialDDFB(full_shape, params["n_layers"], params["n_features"])
        case ("ddfb", True):
            denoiser = DistributedDDFB(
                ctx.comm,
                grid_shape,
                full_shape,
                params["n_layers"],
                params["n_features"],
            )
        case ("dncnn", False):
            denoiser = SerialDnCNN(full_shape)
        case ("dncnn", True):
            denoiser = DistributedDnCNN(ctx.comm, grid_shape, full_shape)
        case ("drunet", False):
            denoiser = SerialDRUNet(full_shape)
        case ("drunet", True):
            denoiser = DistributedDRUNet(ctx.comm, grid_shape, full_shape)
            tile_range = denoiser.tile_range
        case _:
            raise ValueError(f"Unknown denoiser type '{params['type']}'.")
    return denoiser, tile_range


def build_mask(
    obs_cfg: dict,
    cartslicer: CartesianCommSlicer,
    ctx: ExecutionContext,
) -> xp.ndarray:
    mask_loss = obs_cfg["mask_loss"]
    data_seed = obs_cfg["seed_data"]

    if ctx.is_mpi:
        if ctx.is_master == 0:
            ss = np.random.SeedSequence(data_seed)
            # spawn off nworkers child SeedSequences to pass to child processes.
            child_seed = ss.spawn(ctx.comm_size)
        else:
            child_seed = None
        seed = ctx.comm.scatter(child_seed, root=0)
    else:
        seed = data_seed
    rng = np.random.default_rng(seed)

    local_size = cartslicer.tile_size

    # generate local mask and operator
    sz = [local_size[k].item() for k in range(local_size.size - 2, local_size.size)]
    mask = rng.random(sz) < (1 - mask_loss)

    return mask


class PnpInpaintingGeometryHook:
    def build_geometry(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        cfg: dict,
        obs_path: Path,
    ) -> PnpInpaintingGeometry:
        obs_cfg = cfg["observations"]
        gt_path = obs_cfg["img_path"]
        gt_shape = read_img_shape(gt_path)
        # dtype = read_dtype(gt_path)

        grid_shape = ctx.generate_grid_shape(len(gt_shape))

        # create slicer object
        grid_size = np.asarray(grid_shape)
        ranknd = get_ranknd(ctx.rank, grid_size)
        cartslicer = CartesianCommSlicer(
            ranknd,
            grid_size,
            np.asarray(gt_shape),
            np.zeros(len(grid_size), dtype=int),
            np.zeros(len(grid_size), dtype=int),
        )

        if obs_path.exists():
            with io_mng.open(obs_path, mode="r") as f:
                mask = io_mng.read_array(
                    f,
                    "mask",
                    source_slice=cartslicer.slice_global_buffer_to_tile,
                )
        else:
            mask = build_mask(obs_cfg, cartslicer, ctx)

        mask = fit_mask_shape(xp.asarray(mask), cartslicer.tile_size)

        D, _ = build_denoiser(
            cfg["parameters"]["denoiser"],
            gt_shape,
            grid_shape,
            ctx,
        )
        H = build_masking_operator(
            mask,
            gt_shape,
            grid_shape,
            ctx,
            cartslicer,
        )
        # TODO: rework access to mpi slicing utilities
        slicer_x = H.cartslicer if ctx.is_mpi else None
        slicer_y = slicer_x
        slice_x = slicer_x.slice_global_buffer_to_tile if slicer_x else None
        slice_y = slicer_y.slice_global_buffer_to_tile if slicer_y else None
        x_shape = gt_shape
        y_shape = gt_shape
        tile_x_shape = tuple(slicer_x.tile_size) if slicer_x else x_shape
        tile_y_shape = tuple(slicer_y.tile_size) if slicer_y else y_shape
        layout_x = Layout(tile_x_shape, x_shape, slice_x)
        layout_y = Layout(tile_y_shape, y_shape, slice_y)

        return PnpInpaintingGeometry(grid_shape, layout_x, layout_y, H, D, mask)


@dataclass
class GaussianInpaintingObs:
    y: xp.ndarray
    n: xp.ndarray
    Hx: xp.ndarray
    x: xp.ndarray
    mask: xp.ndarray
    interpolation: xp.ndarray
    sigma2: float
    isnr: float
    seed_data: int
    comm_size: int
    is_gpu: bool


class GaussianInpaintingObservationsHook:
    def generate_observations(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        cfg: dict,
        geom: PnpInpaintingGeometry,
    ) -> GaussianInpaintingObs:
        obs_cfg = cfg["observations"]
        img_path = obs_cfg["img_path"]

        with io_mng.open(img_path) as f:
            x = io_mng.read_array(f, "x", geom.layout_x.s)

        Hx = geom.H.forward(x)

        seed_data = obs_cfg["seed_data"]
        rng = create_rng(seed_data, ctx)
        isnr = obs_cfg["isnr"]
        sigma2 = compute_sigma2_from_isnr(Hx, isnr, ctx)

        # TODO: rework rng handling
        if ctx.is_gpu:
            n = torch.normal(
                0, sigma2**0.5, size=geom.layout_y.tile, device="cuda", generator=rng
            )
            n = xp.asarray(n, x.dtype)
        else:
            n = sigma2**0.5 * rng.standard_normal(geom.layout_y.tile, x.dtype)

        y = Hx + n

        # NOTE: in distributed setting, only local interpolation (not equivalent to serial interpolation)
        interpolation = interpolate_masked_image_cubic(y, geom.H.mask).clip(0, 1)

        return GaussianInpaintingObs(
            y,
            n,
            Hx,
            x,
            geom.mask,
            interpolation,
            sigma2,
            isnr,
            seed_data,
            ctx.comm_size,
            ctx.is_gpu,
        )

    def save_observations(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        geom: PnpInpaintingGeometry,
        obs: GaussianInpaintingObs,
        obs_path: Path,
    ) -> None:
        with io_mng.open(obs_path, mode="x") as f:
            io_mng.write_array(f, "y", obs.y, geom.layout_y.full, geom.layout_y.s)
            io_mng.write_array(f, "Hx", obs.Hx, geom.layout_y.full, geom.layout_y.s)
            io_mng.write_array(f, "n", obs.n, geom.layout_y.full, geom.layout_y.s)
            io_mng.write_array(f, "x", obs.x, geom.layout_x.full, geom.layout_x.s)
            io_mng.write_array(f, "mask", obs.mask, geom.layout_x.full, geom.layout_x.s)
            io_mng.write_array(
                f,
                "interpolation",
                obs.interpolation,
                geom.layout_x.full,
                geom.layout_x.s,
            )

        with io_mng.open_master_only(obs_path, mode="r+") as f:
            if f is not None:
                obs_dict = {
                    "sigma2": obs.sigma2,
                    "isnr": obs.isnr,
                    "seed_data": obs.seed_data,
                    "comm_size": obs.comm_size,
                    "is_gpu": obs.is_gpu,
                }
                io_mng.write_config(f, obs_dict)

    def load_observations(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        geom: PnpInpaintingGeometry,
        obs_path: Path,
    ) -> GaussianInpaintingObs:
        with io_mng.open(obs_path, mode="r", force_serial=True) as f:
            y = io_mng.read_array(f, "y", geom.layout_y.s)
            Hx = io_mng.read_array(f, "Hx", geom.layout_y.s)
            n = io_mng.read_array(f, "n", geom.layout_y.s)
            x = io_mng.read_array(f, "x", geom.layout_x.s)
            mask = io_mng.read_array(f, "mask", geom.layout_x.s)
            interpolation = io_mng.read_array(f, "interpolation", geom.layout_x.s)
            obs_dict = io_mng.read_config(f)

        return GaussianInpaintingObs(
            y,
            n,
            Hx,
            x,
            mask,
            interpolation,
            obs_dict["sigma2"],
            obs_dict["isnr"],
            obs_dict["seed_data"],
            ctx.comm_size,
            ctx.is_gpu,
        )


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


class GaussianInpaintingPnpMcmcHook:
    def build_estimators(
        self,
        geom: PnpInpaintingGeometry,
        obs: GaussianInpaintingObs,
    ) -> tuple[dict[str, Variable], list[BaseEstimator]]:

        y_var = Variable(
            layout=geom.layout_y,
            name="Y",
            state=obs.y,
            dtype=obs.y.dtype,
        )

        x_var = Variable(
            layout=geom.layout_x,
            name="X",
            dtype=obs.x.dtype,
        )

        variables = {"X": x_var, "Y": y_var}
        estimators: list[BaseEstimator] = [MMSEVar(x_var), CI(x_var, all_samples=True)]

        return variables, estimators

    def build_model(
        self,
        ctx: ExecutionContext,
        cfg: dict,
        geom: PnpInpaintingGeometry,
        obs: GaussianInpaintingObs,
        vars_: dict[str, Variable],
    ) -> BaseModel:

        reg_coef = cfg["parameters"]["reg_coef"]
        denoiser_params = cfg["parameters"]["denoiser"]
        eps = (
            denoiser_params["denoising_level"] ** 2
            if denoiser_params["denoising_level"] is not None
            else obs.sigma2
        )
        L = denoiser_params.get("L", None) or 1.0
        step_size_X, lambda_ = compute_step_sizes_gaussian_inpainting_pnp(
            obs.sigma2,
            reg_coef,
            L,
            eps,
        )

        x_var = vars_["X"]
        x_var.state = obs.interpolation
        y_var = vars_["Y"]

        model_params = GaussianInpaintingPnpParameters(
            sigma2=obs.sigma2, reg_coeff=reg_coef
        )

        PnpULA = GpuPnpULA if ctx.is_gpu else CpuPnpULA

        X = PnpULA(
            var=x_var,
            step_size=step_size_X,
            reg_coef=reg_coef,
            epsilon=obs.sigma2,
            lambda_=lambda_,
        )

        if ctx.is_mpi:
            Model = DistributedGaussianInpaintingPnpModel
        else:
            Model = GaussianInpaintingPnpModel

        return Model(
            model_params,
            geom.H,
            y_var,
            X,
            geom.D,
        )
