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
