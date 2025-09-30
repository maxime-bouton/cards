r"""Implementation of an MMSE estimator derived from the abstract class
:class:`mcmc.estimator.estimatorBuilder.BaseMMSEBuilder`.
"""

from mcmc.backend import xp
from mcmc.estimator.estimator_builder import BaseMMSEBuilder


class mmse_builder(BaseMMSEBuilder):
    def __init__(self, shape):
        super().__init__()
        self.estimator = xp.zeros(shape)

    def reset(self):
        """Set the estimator to zero."""
        self.estimator.fill(0)

    def aggregate_states(self, state: xp.ndarray):
        self.estimator += state

    def build_estimator(self, N: int):
        self.estimator = self.estimator / N


class multi_gpu_mmse_builder:
    def __init__(self, shape):
        self.estimator = xp.zeros(shape)

    def reset(self):
        self.estimator = xp.zeros_like(
            self.estimator
        )  #! this line prevent further factorisation, to be solved

    def aggregate_states(self, state):
        self.estimator += state

    def build_estimator(self, N):
        self.estimator = self.estimator / N
