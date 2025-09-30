from abc import ABC, abstractmethod

from mcmc.backend import xp


class BaseDenoiser(ABC):
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
