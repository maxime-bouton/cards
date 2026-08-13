from pathlib import Path
from typing import Protocol, TypeVar

from cards.core.execution_context import ExecutionContext
from cards.core.geometry_hook import G
from cards.io.io_manager import IOManager

Obs = TypeVar("Obs")


class ObservationsHook(Protocol[G, Obs]):
    def generate_observations(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        cfg: dict,
        geom: G,
    ) -> Obs: ...

    def save_observations(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        geom: G,
        obs: Obs,
        obs_path: Path,
    ) -> None: ...

    def load_observations(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        geom: G,
        obs_path: Path,
    ) -> Obs: ...
