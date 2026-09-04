# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import cards.backend as xp
from cards.analysis.metrics import psnr, snr, ssim
from cards.core.analysis_hook import AnalysisArtifacts, AnalysisResults
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
from cards.utils.utils import expand_shape_left
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
        estimators: list[BaseEstimator] = [MMSEVar(x_var), CI(x_var, all_samples=False)]

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


def slices_obs_deconv(
    ctx: ExecutionContext,
    grid_shape: tuple[int, ...],
    kernel_shape: tuple[int, ...],
) -> tuple[slice, ...]:
    grid_size = np.asarray(grid_shape)
    kernel_size = np.asarray(expand_shape_left(kernel_shape, len(grid_size)))

    left = kernel_size // 2
    right = -(kernel_size // 2)

    return tuple(slice(l or None, r or None) for l, r in zip(left, right))


class GaussianDeconvPnpAnalysisHook:
    def run_analysis(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        cfg: dict,
        geometry: PnpDeconvGeometry,
        obs: GaussianDeconvObs,
        estimators: list[BaseEstimator],
        burnin: int,
        ckpt_dir: Path,
        obs_path: Path,
    ) -> AnalysisResults:
        n_ckpts = int(cfg["sampler"]["n_ckpts"])
        ckpt_size = int(cfg["sampler"]["ckpt_size"])
        n_iter = n_ckpts * ckpt_size

        ckpt_files = sorted(
            ckpt_dir.glob("checkpoint_*.h5"),
            # NOTE: key is important so `10` doesn't come between `1` and `2`
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        all_keys = [k for e in estimators for k in e.declared_keys]
        all_slices = {k: v for e in estimators for k, v in e.slices.items()}

        # NOTE: estimates concatenation is list based to allow Generator usage in the future
        per_ckpt_local = []

        if ctx.is_master:
            potential = xp.zeros(n_iter, dtype=float)
            computation_time = xp.zeros((ctx.comm_size, n_iter), dtype=float)

        for i, f_path in enumerate(ckpt_files):
            with io_mng.open(f_path, "r") as f:
                per_ckpt_local.append(
                    io_mng.read_dict(f, keys=all_keys, slices=all_slices)
                )
            with io_mng.open_master_only(f_path, "r") as f:
                if f is not None:
                    potential[i * ckpt_size : (i + 1) * ckpt_size] = io_mng.read_array(
                        f, "potential"
                    )
                    computation_time[:, i * ckpt_size : (i + 1) * ckpt_size] = (
                        io_mng.read_array(f, "computation_time")
                    )

        if ctx.is_master:
            t_mean = computation_time.mean(axis=1).round(6)
            t_std = computation_time.std(axis=1).round(6)
            t_min = computation_time.min(axis=1).round(6)
            t_max = computation_time.max(axis=1).round(6)

            # NOTE: `tolist()` ensures correct save with json
            time_dict = {
                "mean": t_mean.tolist(),
                "std": t_std.tolist(),
                "min": t_min.tolist(),
                "max": t_max.tolist(),
            }
        else:
            time_dict = None

        reduced_local: dict[str, xp.ndarray] = {}
        full_shapes: dict[str, tuple[int, ...]] = {}
        slices: dict[str, tuple[slice, ...]] = {}
        for estimator in estimators:
            l = [{k: d[k] for k in estimator.declared_keys} for d in per_ckpt_local]
            reduced_local.update(estimator.reduce_checkpoints(l, burnin, ctx))
            full_shapes.update(estimator.global_shapes)
            slices.update(estimator.slices)

        if "X_mmse" in reduced_local and obs.x is not None:
            reduced_local["X_err"] = xp.abs(reduced_local["X_mmse"] - obs.x)
            full_shapes["X_err"] = full_shapes["X_mmse"]
            slices["X_err"] = slices["X_mmse"]

        artifacts = AnalysisArtifacts(
            original=obs.x,
            observations=obs.y,
            reduced=reduced_local,
            global_shapes=full_shapes,
            slices=slices,
            initialisation=None,
            potential=potential if ctx.is_master else None,
            time=time_dict if ctx.is_master else None,
        )

        # HACK: reloading observations is the current only way to have matching shapes
        # for both cropped observations and GT in mpi settings
        crop = slices_obs_deconv(ctx, geometry.grid_shape, geometry.kernel.shape)
        if ctx.is_mpi:
            list_s = []
            for xs, cs in zip(geometry.layout_x.s, crop):
                l = (xs.start or 0) + (cs.start or 0)
                # NOTE: the right is shifted also by the same `start` value
                r = (xs.stop or 0) + (cs.start or 0)
                list_s.append(slice(l or None, r or None))
            with io_mng.open(obs_path) as f:
                y = io_mng.read_array(f, "y", tuple(list_s))
        else:
            y = obs.y[crop]

        m = {"SNR": snr, "PSNR": psnr, "SSIM": ssim}
        metrics = {
            "X": {
                k: round(v(obs.x, reduced_local["X_mmse"], ctx), 2)
                for k, v in m.items()
            },
            "Y": {k: round(v(obs.x, y, ctx), 2) for k, v in m.items()},
        }
        return AnalysisResults(artifacts=artifacts, metrics=metrics)

    def save_results(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        results: AnalysisResults,
        save_path,
    ) -> None:
        global_shapes = results.artifacts.global_shapes
        slices = results.artifacts.slices
        estim_path = save_path / "estim.h5"
        with io_mng.open(estim_path, "w") as f:
            io_mng.write_dict(f, results.artifacts.reduced, global_shapes, slices)

        with io_mng.open_master_only(estim_path, "r+") as f:
            if f is not None:
                io_mng.write_array(f, "potential", results.artifacts.potential)

        io_mng.write_metrics(
            save_path / "metrics.json",
            results.metrics | {"times (s)": results.artifacts.time},
        )

    def visualize_results(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        results: AnalysisResults,
        save_path: Path,
    ) -> None:
        import matplotlib

        # distributed settings require headless execution to avoid GUI issue
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def format_img(img):
            """Format "torch-like" image to standard (H, W) or (H, W, C)."""
            if img is None:
                return None
            # move channel axis to the end if it exists
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.moveaxis(img, 0, -1)
            # squeeze (H, W, 1) to (H, W) for grayscale
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)
            return img

        estim_path = save_path / "estim.h5"

        with io_mng.open_master_only(estim_path) as f:
            if f is not None:
                for key in f:
                    data = f[key][...]
                    if data.ndim < 2:
                        continue

                    is_uncertainty = any(
                        sub in key.lower() for sub in ["var", "std", "ci", "err"]
                    )
                    cmap = "inferno" if is_uncertainty else None

                    if not is_uncertainty:
                        vmin, vmax = 0, 1
                        img = format_img(data)
                        img = np.clip(img, vmin, vmax)
                        plt.imsave(
                            save_path / f"{key}.jpg",
                            img,
                            vmin=vmin,
                            vmax=vmax,
                        )
                    else:
                        H, W = data.shape[-2:]
                        dpi = 100  # 100 pixels per inch anchor

                        # define layout strictly in pixels
                        gap_px = 15  # space between image and colorbar
                        cb_px = 25  # width of the colorbar itself
                        label_px = 60  # space reserved on the right for text labels

                        W_new = W + gap_px + cb_px + label_px

                        for c in range(data.shape[0]):
                            channel_img = data[c]

                            # create figure with exact pixel dimensions
                            fig = plt.figure(figsize=(W_new / dpi, H / dpi), dpi=dpi)

                            # map the image to the exact left portion of the figure
                            # [left, bottom, width, height] as fractions of the figure
                            ax_img = fig.add_axes([0, 0, W / W_new, 1.0])
                            im = ax_img.imshow(channel_img, cmap=cmap)
                            ax_img.axis("off")

                            # add the colorbar
                            cb_left = (W + gap_px) / W_new
                            cb_width = cb_px / W_new

                            # height is 1.0 so it spans top to bottom identically to the image
                            ax_cb = fig.add_axes([cb_left, 0.05, cb_width, 0.90])
                            fig.colorbar(im, cax=ax_cb)

                            # save exactly as is
                            # do not use bbox_inches="tight" or pad to not alter layout
                            fig.savefig(save_path / f"{key}_{c}.jpg", dpi=dpi)
                            plt.close(fig)

        if ctx.is_master and results.artifacts.potential is not None:
            fig, ax = plt.subplots()
            p = results.artifacts.potential
            ax.plot(p.get() if ctx.is_gpu else p)
            ax.set_title("Potential over sampling iteration steps")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Potential")
            ax.grid(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig.savefig(save_path / "potential.pdf", format="pdf", bbox_inches="tight")
            plt.close(fig)
