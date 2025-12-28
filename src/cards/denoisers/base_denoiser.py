"""Abstract base class for denoisers."""

from abc import ABC, abstractmethod
from pathlib import Path

from cards.backend import xp


class BaseDenoiser(ABC):
    def __init__(
        self,
        weights_path=Path(__file__).parents[3] / "data/weights",
    ):
        r"""Constructor of the BaseDenoiser class.

        Parameters
        ----------
        weights_path : str, optional
            The path to the pre-trained weights folder.
        """
        self.weights_path = weights_path

    @abstractmethod
    def __call__(self, input_image: xp.ndarray, sigma: float) -> xp.ndarray:
        """Apply the denoiser to the input image.

        Parameters
        ----------
        input_image: xp.ndarray
            The input image to be denoised.
        sigma: float
            The regularization parameter (typically Gaussian noise standard deviation).

        Returns
        -------
        xp.ndarray
            The denoised image.
        """
        pass


class BaseDistributedDenoiser(BaseDenoiser, ABC):
    @property
    @abstractmethod
    def global_to_tile_slice(self) -> tuple[slice, ...]: ...
