r"""Abstract base classes for image denoising neural networks."""

from abc import ABC, abstractmethod
from pathlib import Path

import cards.backend as xp


class BaseDenoiser(ABC):
    r"""Generic abstract class encoding serial image denoising neural-networks.

    Parameters
    ----------
    weights_path : str, optional
        Path to the folder containing the pre-trained denoiser weights.

    Attributes
    ----------
    weights_path : str
        Path to the folder containing the pre-trained denoiser weights.

    Methods
    -------
    __call__(self, input_image: xp.ndarray, sigma: float)
       Apply the denoiser to an input image.
    get_recv_size()
        Returns the extent of the ghost cells to be received from neighbour processes along each axis of the Cartesian grid.
    get_send_size()
        Returns the extent of the ghost cells to be sent to neighbour processes along each axis of the Cartesian grid.
    """

    def __init__(
        self,
        weights_path=Path(__file__).parents[3] / "data/weights",
    ):
        self.weights_path = weights_path

    @abstractmethod
    def __call__(self, input_image: xp.ndarray, sigma: float) -> xp.ndarray:
        r"""Apply the denoiser to the input image.

        Parameters
        ----------
        input_image: xp.ndarray
            Input image to be denoised.
        sigma: float
            Denoiser parameter (typically Gaussian noise standard deviation).

        Returns
        -------
        xp.ndarray
            Denoised image.
        """
        pass


class BaseDistributedDenoiser(BaseDenoiser, ABC):
    r"""Generic abatrsct class for distributed image denoising neural-networks.

    Methods
    -------
    global_to_tile_slice()
        Returns slices to extract the image tile handled by the current worker from the internal buffer including ghost cells.
    """

    @property
    @abstractmethod
    def global_to_tile_slice(self) -> tuple[slice, ...]:
        r"""Returns slices to extract the image tile handled by the current worker from the internal buffer including ghost cells."""

    @property
    @abstractmethod
    def get_recv_size(self) -> xp.ndarray:
        r""" "Returns the extent of the ghost cells to be received from neighbour processes along each axis of the Cartesian grid."""

    @property
    @abstractmethod
    def get_send_size(self) -> xp.ndarray:
        r""" "Returns the extent of the ghost cells to be sent to neighbour processes along each axis of the Cartesian grid."""
