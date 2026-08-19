"""Abstract linear operator class."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from abc import ABC, abstractmethod


class LinearOperator(ABC):
    r"""Base model class gathering the parameters of the measurement operator
    underlying the inverse problem to be solved.

    Attributes
    ----------
    image_size : cards.backend.xp.ndarray of int, of size ``d``
        Full image size.
    data_size : cards.backend.xp.ndarray of int, of size ``d``
        Full data size.
    ndims : int
        Number of axis (dimensions) in the problem.
    """

    def __init__(
        self,
        image_size,
        data_size,
    ):
        """LinearOperator constructor.

        Parameters
        ----------
        image_size : cards.backend.xp.ndarray of int, of size ``d``
            Full image size.
        data_size : cards.backend.xp.ndarray of int
            Full data size.
        """
        # if not image_size.size == data_size.size:
        #     raise ValueError(
        #         "image_size and data_size must have the same number of elements"
        #     )
        self.image_size = image_size
        self.data_size = data_size
        self.ndims = image_size.size

    @abstractmethod
    def forward(self, input_image):  # pragma: no cover
        r"""Implementation of the direct operator to update the input array
        ``input_image`` (from image to data space).

        Parameters
        ----------
        input_image : cards.backend.xp.ndarray
            Input array (image space).

        Note
        ----
        The method needs to be implemented in any class inheriting from
        BaseCommunicator.
        """
        pass

    @abstractmethod
    def adjoint(self, input_data):  # pragma: no cover
        r"""Implementation of the adjoint operator to update the input array
        ``input_data`` (from data to image space).

        Parameters
        ----------
        input_data : cards.backend.xp.ndarray
            Input array (data space).

        Note
        ----
        The method needs to be implemented in any class inheriting from
        BaseCommunicator.
        """
        pass
