from typing import Protocol

from cards.core.execution_context import ExecutionContext
from cards.core.geometry_hook import G
from cards.core.observations_hook import Obs
from cards.models import BaseDistributedModel, BaseModel


class McmcHook(Protocol[G, Obs]):
    def build_model(
        self,
        ctx: ExecutionContext,
        cfg: dict,
        geom: G,
        obs: Obs,
    ) -> BaseModel | BaseDistributedModel: ...
