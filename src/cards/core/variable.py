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
