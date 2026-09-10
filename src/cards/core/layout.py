# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)


class Layout:
    full: tuple[int, ...]

    def __init__(
        self,
        tile: tuple[int, ...],
        full: tuple[int, ...] | None = None,
        s: tuple[slice, ...] | None = None,
    ):
        self.tile = tile
        self.full = full if full is not None else tile
        self.s = s
