r"""Serial denoiser class for the DRUNet network :cite:p:`Zhang2021`."""

from pathlib import Path

import numpy as np
import torch

import cards.backend as xp
from cards.denoisers.base_denoiser import BaseDenoiser
from cards.denoisers.denoiser_loader import load_pretrained_drunet
from cards.utils.utils import torch2xp, xp2torch


class SerialDRUNet(BaseDenoiser):
    def __init__(
        self,
        image_size: np.ndarray,
        weights_path=Path(__file__).parents[3] / "data/weights/drunet",
    ):
        r"""Serial DRUNet network :cite:p:`Zhang2021`.

        Parameters
        ----------
        image_size: xp.ndarray
            Input image shape.
        weights_path : str, optional
            Path to the folder containing the pre-trained denoiser weights.
        """
        super(SerialDRUNet, self).__init__(weights_path)
        if image_size.size < 3:
            # NOTE: accommodate gray scale images (implicitly, number of channes is 1)
            n_channels = 1
        else:
            n_channels = image_size[-3]

        self.drunet = load_pretrained_drunet(
            n_channels,
            weights_path=self.weights_path,
        )

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
        # TODO: add error or warning when the number of channels in the input does not fit that of the denoiser
        with torch.no_grad():
            return torch2xp(
                self.drunet(xp2torch(input_image, torch_dtype=torch_dtype), sigma),
                cp_dtype=cp_dtype,
            )
