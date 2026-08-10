from typing import Protocol, TypeVar

from cards.core.execution_context import ExecutionContext

G = TypeVar("G")


class GeometryHook(Protocol[G]):
    def build_geometry(self, cfg: dict, ctx: ExecutionContext) -> G: ...
