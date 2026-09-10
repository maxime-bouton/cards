from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cards.core.validation import SimulationConfig


@dataclass
class PathsHook:
    fn_obs_rel_path: Callable[[SimulationConfig], Path | str] | None = None
    fn_ckpt_rel_path: Callable[[SimulationConfig], Path | str] | None = None
