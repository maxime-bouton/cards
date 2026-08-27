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
from cards.estimators.base_estimator import BaseEstimator
from cards.estimators.ci import CI
from cards.estimators.mmse_var import MMSEVar
from cards.io.io_manager import IOManager
from cards.models import (
    BaseModel,
    DistributedGaussianDeconvolutionTvModel,
    GaussianDeconvolutionTvModel,
)
from cards.models.gaussian_deconvolution_tv_model import GaussianDeconvolutionTvParams
from cards.operators.dft_convolution import DftConvolution
from cards.operators.distributed_dft_convolution import DistributedDftConvolution
from cards.operators.distributed_gradient import DistributedGradient2d
from cards.operators.gradient import Gradient2d
from cards.random import create_rng
from cards.transition_kernels.psgla import CpuPSGLA, GpuPSGLA
from cards.utils.utils_img import read_dtype, read_img_shape
from cards.utils.utils_observations import (
    compute_sigma2_from_isnr,
    fit_kernel_shape,
    generate_gaussian_kernel,
    generate_motion_kernel,
    load_img,
)


@dataclass
class TvDeconvGeometry:
    grid_shape: tuple[int, ...]
    layout_x: Layout
    layout_y: Layout
    layout_z: Layout  # NOTE: see if this is needed or not
    H: DftConvolution | DistributedDftConvolution
    G: Gradient2d | DistributedGradient2d
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


def build_gradient(
    full_shape: tuple[int, ...],
    grid_shape: tuple[int, ...],
    ctx: ExecutionContext,
) -> Gradient2d | DistributedGradient2d:
    if ctx.is_mpi:
        gradient = DistributedGradient2d(full_shape, grid_shape, ctx.comm)
    else:
        gradient = Gradient2d(full_shape)
    return gradient


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


class TvDeconvGeometryHook:
    def build_geometry(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        cfg: dict,
        obs_path: Path,
    ) -> TvDeconvGeometry:
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
        G = build_gradient(
            gt_shape,
            grid_shape,
            ctx,
        )
        H = build_convolution_operator(
            kernel,
            gt_shape,
            grid_shape,
            ctx,
        )
        # TODO: rework access to mpi slicing utilities
        slicer_x = H.direct_communicator.cartslicer if ctx.is_mpi else None
        slicer_y = H.adjoint_communicator.cartslicer if ctx.is_mpi else None
        slice_x = slicer_x.slice_global_buffer_to_tile if slicer_x else None
        slice_y = slicer_y.slice_global_buffer_to_tile if slicer_y else None
        slice_z = G.slice_adjoint_global_buffer_to_tile if ctx.is_mpi else None
        x_shape = gt_shape
        y_shape = tuple(H.data_shape)
        z_shape = tuple(G.data_shape)
        tile_x_shape = tuple(slicer_x.tile_size) if slicer_x else x_shape
        tile_y_shape = tuple(slicer_y.tile_size) if slicer_y else y_shape
        tile_z_shape = (*G.adjoint_tile_size,) if ctx.is_mpi else None
        layout_x = Layout(tile_x_shape, x_shape, slice_x)
        layout_y = Layout(tile_y_shape, y_shape, slice_y)
        layout_z = Layout(tile_z_shape, z_shape, slice_z)
        return TvDeconvGeometry(
            grid_shape, layout_x, layout_y, layout_z, H, G, kernel_2d
        )


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
        geom: TvDeconvGeometry,
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
        geom: TvDeconvGeometry,
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
        geom: TvDeconvGeometry,
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


def compute_step_sizes_gaussian_deconvolution_tv(
    sigma2: float,
    kernel: xp.ndarray,
    split_coef: float,
) -> tuple[float, float]:
    step_size_X = 0.99 / (
        8.0 / split_coef + xp.max(xp.abs(xp.fft.rfft2(kernel))) ** 2 / sigma2
    )
    step_size_Z = 0.99 * split_coef
    return step_size_X, step_size_Z


class GaussianDeconvPnpMcmcHook:
    def build_model(
        self,
        ctx: ExecutionContext,
        cfg: dict,
        geom: TvDeconvGeometry,
        obs: GaussianDeconvObs,
    ) -> tuple[BaseModel, list[BaseEstimator]]:

        reg_coef = cfg["parameters"]["reg_coef"]
        split_coef = cfg["parameters"]["split_coef"]
        step_size_X, step_size_Z = compute_step_sizes_gaussian_deconvolution_tv(
            obs.sigma2,
            obs.kernel,
            split_coef,
        )

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

        z_var = Variable(
            layout=geom.layout_z,
            name="Z",
            dtype=obs.x.dtype,
        )

        model_params = GaussianDeconvolutionTvParams(obs.sigma2, reg_coef, split_coef)

        PSGLA = GpuPSGLA if ctx.is_gpu else CpuPSGLA

        X = PSGLA(
            var=x_var,
            step_size=step_size_X,
        )

        Z = PSGLA(
            var=z_var,
            step_size=step_size_Z,
        )

        if ctx.is_mpi:
            Model = DistributedGaussianDeconvolutionTvModel
        else:
            Model = GaussianDeconvolutionTvModel

        model = Model(
            params=model_params,
            convolution_operator=geom.H,
            gradient_operator=geom.G,
            y=y_var,
            X=X,
            Z=Z,
        )

        estimators: list[BaseEstimator] = [MMSEVar(x_var), CI(x_var, all_samples=True)]

        return model, estimators
