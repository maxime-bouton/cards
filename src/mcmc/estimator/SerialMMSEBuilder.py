r"""NumPy implementation of :class:`mcmc.estimator.estimatorBuilder.BaseMMSEBuilder`."""

import numpy as np
from mcmc.estimator.estimatorBuilder import BaseMMSEBuilder


class SerialMMSEBuilder(BaseMMSEBuilder):
    r"""Numpy implementation of a MMSE estimator.

    Numpy implementation of :class:`mcmc.estimator.estimatorBuilder.BaseMMSEBuilder`.

    Attributes
    ----------
    estimator : np.ndarray
        Internal state of the MMSE estimator.
    """

    def __init__(self, shape) -> None:
        r"""Initializing the MMSE estimator.

        Parameters
        ----------
        shape : tuple[int]
            Shape of the MMSE estimator.
        """
        super(SerialMMSEBuilder, self).__init__()
        self.estimator = np.zeros(shape)

    def reset(self) -> None:
        """Set the internal state of the estimator to 0."""
        self.estimator = np.zeros_like(self.estimator)
