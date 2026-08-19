from typing import Protocol

from cards.core.execution_context import ExecutionContext
from cards.core.geometry_hook import G
from cards.core.observations_hook import Obs
from cards.estimators.base_estimator import BaseEstimator
from cards.models import BaseModel


class McmcHook(Protocol[G, Obs]):
    def build_model(
        self,
        ctx: ExecutionContext,
        cfg: dict,
        geom: G,
        obs: Obs,
    ) -> tuple[BaseModel, list[BaseEstimator]]: ...
