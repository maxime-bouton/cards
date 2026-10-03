from pathlib import Path

import cards.backend as xp
from cards.analysis.metrics import psnr, snr, ssim
from cards.analysis.plotting import (
    plot_potential,
    save_point_estimate,
    save_uncertainty_maps,
)
from cards.analysis.utils import (
    add_error_maps,
    compute_metrics,
    list_checkpoints,
    reduce_all,
    summarize_time,
)
from cards.core.execution_context import ExecutionContext
from cards.core.validation import SimulationConfig
from cards.estimators.base_estimator import BaseEstimator
from cards.hooks.analysis_hook import AnalysisArtifacts, AnalysisHook, AnalysisResults
from cards.io.io_manager import IOManager


class DefaultAnalysisHook[G, O](AnalysisHook[G, O]):
    def prepare_metrics_data(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        geometry: G,
        obs: O,
        reduced_local: dict[str, xp.ndarray],
        obs_path: Path,
    ) -> tuple[dict[str, xp.ndarray], dict[str, xp.ndarray]]:
        """Override in subclass to carry out domain-specific actions (e.g. cropping)."""
        raise NotImplementedError

    def run_analysis(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        cfg: SimulationConfig,
        geometry: G,
        obs: O,
        estimators: list[BaseEstimator],
        burnin: int,
        ckpt_dir: Path,
        obs_path: Path,
    ) -> AnalysisResults:

        n_ckpts, ckpt_size = int(cfg.sampler.n_ckpts), int(cfg.sampler.ckpt_size)
        n_iter = n_ckpts * ckpt_size
        ckpt_files = list_checkpoints(ckpt_dir, cfg.io.ckpt_prefix, n_ckpts)

        all_keys = [k for e in estimators for k in e.declared_keys]
        all_slices = {k: v for e in estimators for k, v in e.slices.items()}
        per_ckpt_local = []

        for i, f_path in enumerate(ckpt_files):
            with io_mng.open(f_path, "r") as f:
                if i == 0 and ctx.is_master:
                    potential = xp.zeros(n_iter, dtype=float)
                    comm_size = io_mng.stacked_size(f, "computation_time")
                    computation_time = xp.zeros((comm_size, n_iter), dtype=float)
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

        time_dict = summarize_time(computation_time) if ctx.is_master else None

        reduced_local, full_shapes, slices, uncertainty_keys = reduce_all(
            estimators, per_ckpt_local, burnin, ctx
        )

        ground_truth = {"X": getattr(obs, "x", None)}
        add_error_maps(reduced_local, full_shapes, slices, ground_truth)

        uncertainty_keys.update(k for k in reduced_local if k.endswith("_err"))

        targets, references = self.prepare_metrics_data(
            ctx, io_mng, geometry, obs, reduced_local, obs_path
        )
        metrics = compute_metrics(
            targets, references, {"SNR": snr, "PSNR": psnr, "SSIM": ssim}, ctx
        )

        artifacts = AnalysisArtifacts(
            original=getattr(obs, "x", None),
            observations=getattr(obs, "y", None),
            reduced=reduced_local,
            global_shapes=full_shapes,
            slices=slices,
            initialisation=None,
            potential=potential if ctx.is_master else None,
            time=time_dict if ctx.is_master else None,
            uncertainty_keys=uncertainty_keys,
        )

        return AnalysisResults(artifacts=artifacts, metrics=metrics)

    def save_results(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        results: AnalysisResults,
        save_path: Path,
    ) -> None:
        estim_path = save_path / "estim.h5"
        with io_mng.open(estim_path, "w") as f:
            io_mng.write_dict(
                f,
                results.artifacts.reduced,
                results.artifacts.global_shapes,
                results.artifacts.slices,
            )

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
        estim_path = save_path / "estim.h5"

        with io_mng.open_master_only(estim_path) as f:
            if f is not None:
                for key in f:
                    data = f[key][...]
                    if data.ndim < 2 or data.ndim > 3:
                        continue

                    if key in results.artifacts.uncertainty_keys:
                        save_uncertainty_maps(data, save_path / key, cmap="inferno")
                    else:
                        save_point_estimate(data, save_path / f"{key}.jpg")

        if ctx.is_master and results.artifacts.potential is not None:
            plot_potential(results.artifacts.potential, save_path / "potential.pdf")
