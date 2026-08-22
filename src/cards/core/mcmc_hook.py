# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

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
