r"""Abstract base class for denoiser networks."""

from abc import ABC, abstractmethod
from pathlib import Path

import cards.backend as xp


class BaseDenoiser(ABC):
    def __init__(
        self,
        weights_path=Path(__file__).parents[3] / "data/weights",
    ):
        r"""Constructor of the BaseDenoiser class.

        Parameters
        ----------
        weights_path : str, optional
            Path to the folder containing the pre-trained denoiser weights.
        """
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
    @property
    @abstractmethod
    def global_to_tile_slice(self) -> tuple[slice, ...]: ...
