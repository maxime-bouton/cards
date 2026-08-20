"""Abstract linear operator class."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from abc import ABC, abstractmethod
from collections.abc import Sequence

import cards.backend as xp


class LinearOperator(ABC):
    r"""Base model class gathering the parameters of the measurement operator
    underlying the inverse problem to be solved.

    Attributes
    ----------
    image_shape : Sequence[int]
        Full image shape.
    data_shape : Sequence[int]
        Full output data shape.
    ndims : int
        Number of axis (dimensions) in image space.
    ndims_adjoint : int
        Number of axis (dimensions) in data space.
    """

    def __init__(
        self,
        image_shape: Sequence[int],
        data_shape: Sequence[int],
    ):
        self.image_shape = image_shape
        self.data_shape = data_shape
        self.ndims = len(image_shape)
        self.ndims_adjoint = len(data_shape)

    @abstractmethod
    def forward(self, input_image: xp.ndarray) -> xp.ndarray:  # pragma: no cover
        r"""Implementation of the direct operator (from image to data space).

        Parameters
        ----------
        input_image : xp.ndarray
            Input tensor (image space).

        Returns
        -------
        xp.ndarray
            Ouput tensor (data space).
        """

    @abstractmethod
    def adjoint(self, input_data: xp.ndarray) -> xp.ndarray:  # pragma: no cover
        r"""Implementation of the adjoint operator (from data to image space).

        Parameters
        ----------
        input_data : xp.ndarray
            Input tensor (data space).

        Returns
        -------
        xp.ndarray
            Ouput tensor (image space).
        """
