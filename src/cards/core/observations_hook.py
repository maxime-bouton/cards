# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

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
