r"""CuPy implementation of :class:`mcmc.estimator.estimatorBuilder.BaseMMSEBuilder`."""

import cupy as cp
from mcmc.estimator.estimatorBuilder import BaseMMSEBuilder


class GPUMMSEBuilder(BaseMMSEBuilder):
    r"""CuPy implementation of a MMSE estimator.

    CuPy implementation of :class:`mcmc.estimator.estimatorBuilder.BaseMMSEBuilder`.

    Attributes
    ----------
    estimator : cp.ndarray
        Internal state of the MMSE estimator.
    """

    def __init__(self, shape) -> None:
        r"""Initializing the MMSE estimator.

        Parameters
        ----------
        shape : tuple[int]
            Shape of the MMSE estimator.
        """
        super(GPUMMSEBuilder, self).__init__()
        self.estimator = cp.zeros(shape)

    def reset(self) -> None:
        """Set the internal state of the estimator to 0."""
        self.estimator = cp.zeros_like(self.estimator)

class MultiGpuMMSEBuilder(BaseMMSEBuilder):
    def __init__(self, shape, rank)->None:
        super().__init__()
        self.rank = rank
        with cp.cuda.Device(self.rank):
            self.estimator = cp.zeros(shape)
    
    def device_reset(self,rank):
        with cp.cuda.Device(self.rank):
            self.estimator = cp.zeros_like(self.estimator)
    
    def reset(self):
        self.device_reset(self.rank)
    
    def aggregate_states(self, state : cp.ndarray ):
        with cp.cuda.Device(self.rank):
            self.estimator += state

    def build_estimator(self, N : int):
        with cp.cuda.Device(self.rank):
            self.estimator = self.estimator / N
