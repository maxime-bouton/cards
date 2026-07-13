r"""Abstract base classes for image denoising neural networks."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch

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
    """

    def __init__(
        self,
        weights_path=Path(__file__).parents[3] / "data/weights",
    ):
        self.weights_path = weights_path

    @abstractmethod
    def __call__(
        self,
        input_image: xp.ndarray,
        sigma: float,
        torch_dtype: xp.dtype | None = None,
        cp_dtype: torch.dtype | None = None,
    ) -> xp.ndarray:
        r"""Apply the serial denoiser.

        Parameters
        ----------
        input_image: xp.ndarray
            Input image tile.
        sigma: float
            Denoiser parameter (noise standard deviation).
        torch_dtype : torch.dtype or None, optional
            Numerical precision to be used for computations with `torch`. Default is `None`.
        cp_dtype : xp.dtype or None, optional
            Numerical precision to be used for computations with `xp` (`numpy`
            or `cupy`). Default is `None`.

        Returns
        -------
        xp.ndarray
            Denoised image.
        """


class BaseDistributedDenoiser(BaseDenoiser, ABC):
    r"""Generic abatrsct class for distributed image denoising neural-networks."""

    @property
    @abstractmethod
    def global_to_tile_slice(self) -> tuple[slice, ...]:
        r"""Returns slices to extract the image tile handled by the current worker from the internal buffer including ghost cells."""

    @property
    @abstractmethod
    def get_recv_size(self) -> xp.ndarray:
        r"""Returns the extent of the ghost cells to be received from neighbour processes along each axis of the Cartesian grid."""

    @property
    @abstractmethod
    def get_send_size(self) -> xp.ndarray:
        r"""Returns the extent of the ghost cells to be sent to neighbour processes along each axis of the Cartesian grid."""
