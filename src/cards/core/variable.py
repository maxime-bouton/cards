# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

import cards.backend as xp
from cards.core.layout import Layout


class Variable:
    state: xp.ndarray

    def __init__(
        self,
        layout: Layout,
        name: str = "X",
        state: xp.ndarray | None = None,
        dtype: xp.dtype | None = None,
    ) -> None:
        self.name = name
        self.layout = layout
        self.state = xp.zeros(layout.tile, dtype=dtype) if state is None else state
