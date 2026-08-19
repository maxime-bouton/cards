"""Utility class to handle dynamic path management for CARDS applications."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from collections.abc import Callable
from pathlib import Path
from typing import Any

from cards.core.execution_context import ExecutionContext

DEFAULT_ROOT_DIR_PATH = Path.cwd() / "produced_data"
DEFAULT_PROBLEM_NAME = "inverse_problem"
DEFAULT_APPLICATION_NAME = "application"
DEFAULT_OBS_FILE_STEM = "data"
DEFAULT_CKPT_PREFIX = "checkpoint_"
DEFAULT_LOG_PREFIX = "sampling"
DEFAULT_CKPT_SIZE = 100
DEFAULT_SEED = 42


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
        cfg: dict,
        ctx: ExecutionContext,
        fn_obs_rel_path: Callable[[dict], Path | str] | None = None,
        fn_ckpt_rel_path: Callable[[dict], Path | str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.ctx = ctx
        self.fn_obs_rel_path = fn_obs_rel_path
        self.fn_ckpt_rel_path = fn_ckpt_rel_path

        self.app = self.cfg.get("application", {})
        self.io = self.cfg.get("io", {})

    def get_obs_dir(self) -> Path:
        json_obs_path = self.io.get("obs_dir_path")
        if json_obs_path:
            return Path(json_obs_path)

        root_dir = self.io.get("root_dir_path")
        if not root_dir:
            root_dir = DEFAULT_ROOT_DIR_PATH

        path = Path(root_dir) / self.app.get("type", DEFAULT_PROBLEM_NAME)

        if self.fn_obs_rel_path:
            path /= self.fn_obs_rel_path(self.cfg)

        return path

    def get_obs_path(self) -> Path:
        file = self.io.get("obs_file_stem", DEFAULT_OBS_FILE_STEM) + ".h5"
        return self.get_obs_dir() / file

    def get_ckpt_dir(self) -> Path:
        json_ckpt_path = self.io.get("ckpt_dir_path")
        if json_ckpt_path:
            return Path(json_ckpt_path) / str(self.ctx)

        path = self.get_obs_dir() / self.app.get("name", DEFAULT_APPLICATION_NAME)

        if self.fn_ckpt_rel_path:
            path /= self.fn_ckpt_rel_path(self.cfg)

        sampler = self.cfg.get("sampler", {})
        ckpt_size = sampler.get("ckpt_size", DEFAULT_CKPT_SIZE)
        seed = sampler.get("seed", DEFAULT_SEED)

        return path / f"ckpt_size{ckpt_size}_seed{seed}" / str(self.ctx)

    def get_log_path(self) -> Path:
        json_log_path = self.io.get("log_file_path")
        if json_log_path:
            return Path(json_log_path)

        log_stem = self.io.get("log_file_prefix", DEFAULT_LOG_PREFIX)
        if self.ctx.is_mpi:
            log_stem += f"_{self.ctx.rank}"

        return self.get_ckpt_dir() / f"{log_stem}.log"
