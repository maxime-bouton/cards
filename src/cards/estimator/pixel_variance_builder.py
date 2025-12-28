r"""Implementation of the MMSE and pixel variance estimates using Welford's online algorithm."""

from cards.backend import xp
from cards.estimator.estimator_builder import BaseEstimatorBuilder

# TODO: verify compatibility of interface with other EstimatorBuilder
# _estimator, dictionary, ...


class PixelVarianceBuilder(BaseEstimatorBuilder):
    """Combined MMSE and pixel-wise variance estimator using Welford's online algorithm.

    Efficiently computes both mean (MMSE) and variance estimates simultaneously
    with minimal memory overhead.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: xp.dtype | None = None,
        name: str = "X",
    ):
        super().__init__(shape, dtype=dtype, name=name)
        self._count = 0
        self._mean = xp.zeros(shape, dtype=dtype)
        self._m2 = xp.zeros(shape, dtype=dtype)
        self._name = name

    def aggregate_states(self, state: xp.ndarray):
        """Update running statistics with new samples using Welford's algorithm.

        Parameters
        ----------
        state : xp.ndarray
            New samples to incorporate into estimates.
        """
        self._count += 1
        delta = state - self._mean
        self._mean += delta / self._count
        delta2 = state - self._mean
        self._m2 += delta * delta2

    def build_estimator(self):
        """Finalize both mean and variance computations."""
        self._estimator = {
            f"{self._name}_mmse": self._mean,
            f"{self._name}_var": self._m2 / (self._count - 1),
        }

    def reset(self):
        self._count = 0
        self._mean.fill(0)
        self._m2.fill(0)

    @property
    def estimator(self) -> dict[str, xp.ndarray]:
        return self._estimator
