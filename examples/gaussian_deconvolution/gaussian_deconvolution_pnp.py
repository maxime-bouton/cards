# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import cards.backend as xp
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
    DistributedGaussianDeconvolutionPnpModel,
    GaussianDeconvolutionPnpModel,
)
from cards.models.gaussian_deconvolution_pnp_model import GaussianDeconvolutionPnpParams
from cards.operators.dft_convolution import DftConvolution
from cards.operators.distributed_dft_convolution import DistributedDftConvolution
from cards.random import create_rng
from cards.transition_kernels.pnp_ula import CpuPnpULA, GpuPnpULA
from cards.utils.utils_img import read_dtype, read_img_shape
from cards.utils.utils_observations import (
    compute_sigma2_from_isnr,
    fit_kernel_shape,
    generate_gaussian_kernel,
    generate_motion_kernel,
    load_img,
)


@dataclass
class PnpDeconvGeometry:
    grid_shape: tuple[int, ...]
    layout_x: Layout
    layout_y: Layout
    H: DftConvolution | DistributedDftConvolution
    D: BaseDenoiser
    kernel: xp.ndarray


def build_convolution_operator(
    kernel: xp.ndarray,
    full_shape: tuple[int, ...],
    grid_shape: tuple[int, ...],
    ctx: ExecutionContext,
    tile_range: np.ndarray | None = None,
) -> DftConvolution | DistributedDftConvolution:
    if ctx.is_mpi:
        return DistributedDftConvolution(
            full_shape,
            grid_shape,
            ctx.comm,
            kernel,
            tile_range=tile_range,
        )
    data_size = np.asarray(full_shape) + np.asarray(kernel.shape) - 1
    return DftConvolution(full_shape, tuple(data_size), kernel)


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


def build_kernel(obs_cfg: dict, dtype: type | None = None) -> xp.ndarray:
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
    def build_geometry(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        cfg: dict,
        obs_path: Path,
    ) -> PnpDeconvGeometry:
        obs_cfg = cfg["observations"]
        gt_path = obs_cfg["img_path"]
        gt_shape = read_img_shape(gt_path)
        dtype = read_dtype(gt_path)

        grid_shape = ctx.generate_grid_shape(len(gt_shape))

        if obs_path.exists():
            with io_mng.open(obs_path, mode="r", force_serial=True) as f:
                kernel_2d = io_mng.read_array(f, "kernel")
        else:
            kernel_2d = build_kernel(obs_cfg, dtype)

        kernel = fit_kernel_shape(kernel_2d, gt_shape)
        D, tile_range = build_denoiser(
            cfg["parameters"]["denoiser"],
            gt_shape,
            grid_shape,
            ctx,
        )
        H = build_convolution_operator(
            kernel,
            gt_shape,
            grid_shape,
            ctx,
            tile_range=tile_range,
        )
        # TODO: rework access to mpi slicing utilities
        slicer_x = H.direct_communicator.cartslicer if ctx.is_mpi else None
        slicer_y = H.adjoint_communicator.cartslicer if ctx.is_mpi else None
        slice_x = slicer_x.slice_global_buffer_to_tile if slicer_x else None
        slice_y = slicer_y.slice_global_buffer_to_tile if slicer_y else None
        x_shape = gt_shape
        y_shape = tuple(H.data_shape)
        tile_x_shape = tuple(slicer_x.tile_size) if slicer_x else x_shape
        tile_y_shape = tuple(slicer_y.tile_size) if slicer_y else y_shape
        layout_x = Layout(tile_x_shape, x_shape, slice_x)
        layout_y = Layout(tile_y_shape, y_shape, slice_y)
        return PnpDeconvGeometry(grid_shape, layout_x, layout_y, H, D, kernel_2d)


@dataclass
class GaussianDeconvObs:
    y: xp.ndarray
    n: xp.ndarray
    Hx: xp.ndarray
    x: xp.ndarray
    kernel: xp.ndarray
    sigma2: float
    isnr: float
    seed_data: int
    comm_size: int
    is_gpu: bool


class GaussianDeconvObservationsHook:
    def generate_observations(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        cfg: dict,
        geom: PnpDeconvGeometry,
    ) -> GaussianDeconvObs:
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
        return GaussianDeconvObs(
            y, n, Hx, x, geom.kernel, sigma2, isnr, seed_data, ctx.comm_size, ctx.is_gpu
        )

    def save_observations(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        geom: PnpDeconvGeometry,
        obs: GaussianDeconvObs,
        obs_path: Path,
    ) -> None:
        with io_mng.open(obs_path, mode="x") as f:
            io_mng.write_array(f, "y", obs.y, geom.layout_y.full, geom.layout_y.s)
            io_mng.write_array(f, "Hx", obs.Hx, geom.layout_y.full, geom.layout_y.s)
            io_mng.write_array(f, "n", obs.n, geom.layout_y.full, geom.layout_y.s)
            io_mng.write_array(f, "x", obs.x, geom.layout_x.full, geom.layout_x.s)

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
                io_mng.write_array(f, "kernel", obs.kernel)

    def load_observations(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        geom: PnpDeconvGeometry,
        obs_path: Path,
    ) -> GaussianDeconvObs:
        with io_mng.open(obs_path, mode="r", force_serial=True) as f:
            y = io_mng.read_array(f, "y", geom.layout_y.s)
            Hx = io_mng.read_array(f, "Hx", geom.layout_y.s)
            n = io_mng.read_array(f, "n", geom.layout_y.s)
            x = io_mng.read_array(f, "x", geom.layout_x.s)
            kernel = io_mng.read_array(f, "kernel")
            obs_dict = io_mng.read_config(f)

        return GaussianDeconvObs(
            y,
            n,
            Hx,
            x,
            kernel,
            obs_dict["sigma2"],
            obs_dict["isnr"],
            obs_dict["seed_data"],
            ctx.comm_size,
            ctx.is_gpu,
        )


def compute_step_sizes_gaussian_deconvolution_pnp(
    sigma2: float,
    kernel: xp.ndarray,
    reg_coef: float,
    L: float,
    eps: float,
) -> tuple[float, float]:
    eps = eps or sigma2
    Ly = float(xp.max(xp.abs(xp.fft.rfft2(kernel)))) ** 2 / sigma2
    lambda_ = 0.99 / (2 * L / eps + 4 * Ly)
    be = (reg_coef * L) / eps + 1 / lambda_ + Ly
    step_size_X = 0.99 / (3 * be)
    return step_size_X, lambda_


class GaussianDeconvPnpMcmcHook:
    def build_estimators(
        self,
        geom: PnpDeconvGeometry,
        obs: GaussianDeconvObs,
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
        geom: PnpDeconvGeometry,
        obs: GaussianDeconvObs,
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
        step_size_X, lambda_ = compute_step_sizes_gaussian_deconvolution_pnp(
            obs.sigma2,
            obs.kernel,
            reg_coef,
            L,
            eps,
        )

        x_var = vars_["X"]
        y_var = vars_["Y"]

        model_params = GaussianDeconvolutionPnpParams(
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
            Model = DistributedGaussianDeconvolutionPnpModel
        else:
            Model = GaussianDeconvolutionPnpModel

        return Model(
            params=model_params,
            convolution_operator=geom.H,
            y=y_var,
            X=X,
            denoiser=geom.D,
        )
