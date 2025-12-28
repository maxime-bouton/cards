r"""Implementation of the MMSE estimate using an online computing approach."""

from cards.backend import xp
from cards.estimator.estimator_builder import BaseEstimatorBuilder


class MMSEBuilder(BaseEstimatorBuilder):
    r"""Implementation of an MMSE estimator.

    Computation is carried out online.
    """

    def __init__(
        self, shape: tuple[int, ...], dtype: xp.dtype | None = None, name: str = "X"
    ):
        super().__init__(shape, dtype=dtype, name=name)
        self._name += "_mmse"
        self.estimator = xp.zeros(shape, dtype=dtype)

    def aggregate_states(self, state: xp.ndarray) -> None:
        self.estimator += state

    def build_estimator(self, N: int) -> None:
        self.estimator /= N

    def reset(self):
        self.estimator.fill(0)
