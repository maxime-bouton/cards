r"""Class desribing the way variables encoded by nd-arrays are tessellated across multiple workers (referred to as tiled in the following, or sharding in other libraries)."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

# TODO: check documentation


class Layout:
    r"""Description of the tessellation strategy adopted for nd-arrays..

    Attributes
    ----------
    tile : tuple[int, ...]
        Shape of the nd-array tile owned by the current process.
    full : tuple[int, ...] | None, optional
        Shape of the full nd-array.
    s : tuple[slice, ...] | None, optional
        Slice to extract the array tile for the current process from the full array, by default None
    """

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
