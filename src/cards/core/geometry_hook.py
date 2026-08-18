from pathlib import Path
from typing import Protocol, TypeVar

from cards.core.execution_context import ExecutionContext
from cards.io.io_manager import IOManager

G = TypeVar("G")


class GeometryHook(Protocol[G]):
    def build_geometry(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        cfg: dict,
        obs_path: Path,
    ) -> G: ...
