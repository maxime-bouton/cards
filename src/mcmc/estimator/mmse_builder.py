r"""Implementation of an MMSE estimator derived from the abstract class
:class:`mcmc.estimator.estimatorBuilder.BaseMMSEBuilder`.
"""

import cupy as cp

from mcmc.backend import gpu_context, xp
from mcmc.estimator.estimator_builder import BaseMMSEBuilder


class mmse_builder(BaseMMSEBuilder):
    def __init__(self, shape) -> None:
        super().__init__()
        self.estimator = xp.zeros(shape)

    def reset(self):
        """Set the estimator to zero."""
        self.estimator.fill(0)

    def aggregate_states(self, state: cp.ndarray):
        self.estimator += state

    def build_estimator(self, N: int):
        self.estimator = self.estimator / N


class multi_gpu_mmse_builder:
    def __init__(self, shape, gpu_id=0):
        self.gpu_id = gpu_id
        with gpu_context(self.gpu_id):
            self.estimator = xp.zeros(shape)

    def reset(self):
        with gpu_context(self.gpu_id):
            self.estimator = xp.zeros_like(
                self.estimator
            )  #! this line prevent further factorisation, to be solved

    def aggregate_states(self, state):
        with gpu_context(self.gpu_id):
            self.estimator += state

    def build_estimator(self, N):
        with gpu_context(self.gpu_id):
            self.estimator = self.estimator / N
