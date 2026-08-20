# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

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
