"""Implementation of a MMSE builder dervived from an abstarct class."""

from mcmc.backend import xp
from mcmc.estimator.base_estimator_builder import BaseEstimatorBuilder


class MMSEBuilder(BaseEstimatorBuilder):
    """Implementation of a MMSE estimator."""

    def __init__(self, shape, dtype: xp.dtype | None = None, name: str = "X"):
        super().__init__()
        self._count = 0
        self._sum = xp.zeros(shape, dtype=dtype)
        self._name = name

    def aggregate_states(self, state: xp.ndarray):
        self._sum += state

    def build_estimator(self):
        self._estimator = {f"{self._name}_mmse": self._sum / self._count}

    def reset(self):
        self._count = 0
        self._sum.fill(0)

    @property
    def estimator(self):
        return self._estimator
