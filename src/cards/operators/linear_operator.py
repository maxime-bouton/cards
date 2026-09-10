"""Abstract linear operator class."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import cards.backend as xp


class LinearOperator(ABC):
    r"""Base model class gathering the parameters of the measurement operator
    underlying the inverse problem to be solved.

    Parameters
    ----------
    image_shape : Sequence[int]
        Full image shape.
    data_shape : Sequence[int]
        Full output data shape.

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
    def forward(
        self, image: xp.ndarray, op: Any | None = None
    ) -> xp.ndarray:  # pragma: no cover
        r"""Implementation of the direct operator (from image to data space).

        Parameters
        ----------
        image : xp.ndarray
            Input tensor (image space).
        op : Callable[[Any], Any] | None, optional
            Optional callable, by default None.

        Returns
        -------
        xp.ndarray
            Ouput tensor (data space).
        """

    @abstractmethod
    def adjoint(
        self, data: xp.ndarray, adjoint_op: Any | None = None
    ) -> xp.ndarray:  # pragma: no cover
        r"""Implementation of the adjoint operator (from data to image space).

        Parameters
        ----------
        data : xp.ndarray
            Input tensor (data space).
        adjoint_op : Callable[[Any], Any] | None, optional
            Optional callable, by default None.

        Returns
        -------
        xp.ndarray
            Ouput tensor (image space).
        """
