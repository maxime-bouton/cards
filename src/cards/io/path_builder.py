"""Utility class to handle dynamic path management for CARDS applications."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from collections.abc import Callable
from pathlib import Path
from typing import Any

from cards.core.execution_context import ExecutionContext
from cards.core.validation import SimulationConfig


def clean(val: Any) -> str:
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    return str(val).replace(".", "_")


def dict_to_str(params: dict, ignore_keys: list | None = None) -> str:
    ignore_keys = list(ignore_keys) if ignore_keys else []
    components = []

    if "type" in params and "type" not in ignore_keys:
        components.append(str(params["type"]))
        ignore_keys.append("type")

    for k, v in params.items():
        if k not in ignore_keys and isinstance(v, (int, float, str)):
            components.append(f"{k}{clean(v)}")

    return "-".join(components) if components else ""


class PathBuilder:
    def __init__(
        self,
        cfg: SimulationConfig,
        ctx: ExecutionContext,
        fn_obs_rel_path: Callable[[SimulationConfig], Path | str] | None = None,
        fn_ckpt_rel_path: Callable[[SimulationConfig], Path | str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.ctx = ctx
        self.fn_obs_rel_path = fn_obs_rel_path
        self.fn_ckpt_rel_path = fn_ckpt_rel_path

        src_ctx = self.cfg.analysis.source_context
        self.src_tag = src_ctx if src_ctx is not None else ctx.tag

        self.app = cfg.application
        self.io = cfg.io

    @property
    def _is_cross_context(self) -> bool:
        return self.src_tag != self.ctx.tag

    def get_obs_dir(self) -> Path:
        if self.io.obs_dir_path:
            return self.io.obs_dir_path

        path = self.io.root_dir_path / self.app.type

        if self.fn_obs_rel_path:
            path /= self.fn_obs_rel_path(self.cfg)

        return path

    def get_obs_path(self) -> Path:
        return self.get_obs_dir() / (self.io.obs_file_stem + ".h5")

    def get_ckpt_dir(self) -> Path:
        if self.io.ckpt_dir_path:
            return self.io.ckpt_dir_path / str(self.src_tag)

        path = self.get_obs_dir() / self.app.name
        if self.fn_ckpt_rel_path:
            path /= self.fn_ckpt_rel_path(self.cfg)

        ckpt_size = self.cfg.sampler.ckpt_size
        seed = self.cfg.sampler.seed
        return path / f"ckpt_size{ckpt_size}_seed{seed}" / str(self.src_tag)

    def get_log_path(self) -> Path:
        if self.io.log_file_path:
            return self.io.log_file_path

        log_stem = self.io.log_file_prefix
        if self.ctx.is_mpi:
            log_stem += f"_{self.ctx.rank}"

        if self._is_cross_context:
            ckpt_dir = self.get_analysis_dir()
        else:
            ckpt_dir = self.get_ckpt_dir()
        return ckpt_dir / f"{log_stem}.log"

    def get_analysis_dir(self) -> Path:
        analysis_dir = self.get_ckpt_dir() / f"burnin_{self.cfg.analysis.burnin}"
        if self._is_cross_context:
            analysis_dir = analysis_dir / str(self.ctx)
        return analysis_dir
