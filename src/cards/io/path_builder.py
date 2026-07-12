"""Utility class to handle dynamic path management for CARDS applications."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from cards.core.execution_context import ExecutionContext

DEFAULT_ROOT_DIR_PATH = Path(__file__).parents[3] / "produced_data"
DEFAULT_PROBLEM_NAME = "inverse_problem"
DEFAULT_APPLICATION_NAME = "application"
DEFAULT_OBS_FILE_STEM = "data"
DEFAULT_CKPT_PREFIX = "checkpoint_"
DEFAULT_LOG_PREFIX = "sampling"


def clean(val: Any) -> str:
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    return str(val).replace(".", "_")


def dict_to_str(params: dict, ignore_keys: list | None = None) -> str:
    ignore_keys = ignore_keys or []
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
        config: dict,
        context: ExecutionContext,
        fn_obs_rel_path: Callable[[dict], Path | str] | None = None,
        fn_ckpt_rel_path: Callable[[dict], Path | str] | None = None,
    ) -> None:
        self.config = config
        self.context = context
        self.fn_obs_rel_path = fn_obs_rel_path
        self.fn_ckpt_rel_path = fn_ckpt_rel_path

        self.app = self.config.get("application", {})

        self.io = self.config.get("io", {})

    def get_obs_dir(self) -> Path:
        json_obs_path = self.io.get("obs_dir_path", "")
        if json_obs_path != "":
            return Path(json_obs_path)

        root_dir = self.io.get("root_dir_path", "")
        if root_dir == "":
            root_dir = DEFAULT_ROOT_DIR_PATH
        path = Path(root_dir) / self.app.get("type", DEFAULT_PROBLEM_NAME)

        if self.fn_obs_rel_path:
            path /= self.fn_obs_rel_path(self.config)

        return path

    def get_obs_path(self) -> Path:
        file = self.io.get("obs_file_stem", DEFAULT_OBS_FILE_STEM) + ".h5"
        return self.get_obs_dir() / file

    def get_ckpt_dir(self) -> Path:
        json_ckpt_path = self.io.get("ckpt_dir_path", "")
        if json_ckpt_path != "":
            return Path(json_ckpt_path) / str(self.context)

        path = self.get_obs_dir() / self.app.get("name", DEFAULT_APPLICATION_NAME)

        if self.fn_ckpt_rel_path:
            path /= self.fn_ckpt_rel_path(self.config)

        sampler = self.config["sampler"]
        ckpt_size = sampler["ckpt_size"]
        seed = sampler["seed"]

        return path / f"ckpt_size{ckpt_size}_seed{seed}" / str(self.context)

    def get_log_path(self) -> Path:
        json_log_path = self.io.get("log_file_path", "")
        if json_log_path != "":
            return Path(json_log_path)

        log_stem = self.io.get("log_file_prefix", DEFAULT_LOG_PREFIX)
        if self.context.is_mpi:
            log_stem += f"_{self.context.rank}"
        return self.get_ckpt_dir() / f"{log_stem}.log"
