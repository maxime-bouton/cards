r"""Serial denoiser class for the DnCNN network :cite:`Zhang2017`."""

from pathlib import Path

import numpy as np
import torch

from cards.backend import xp
from cards.denoisers.base_denoiser import BaseDenoiser
from cards.denoisers.denoiser_loader import load_pretrained_dncnn
from cards.utils.utils import torch2xp, xp2torch


class SerialDnCNN(BaseDenoiser):
    def __init__(
        self,
        image_size: np.ndarray,
        weights_path=Path(__file__).parents[3] / "data/weights/dncnn",
    ):
        """
        Serial DnCNN.

        Parameters
        ----------
        image_size: np.ndarray
            The input shape.
        weights_path : str, optional
            The path to the pre-trained weights folder.
        """
        super(SerialDnCNN, self).__init__(weights_path)
        if image_size.size < 3:
            # NOTE: accommodate gray scale images (implicitly, number of channes is 1)
            n_channels = 1
        else:
            n_channels = image_size[-3]

        self.dncnn = load_pretrained_dncnn(n_channels, weights_path=self.weights_path)

    def __call__(self, input_image: xp.ndarray, sigma: float) -> xp.ndarray:
        # TODO: add error or warning when the number of channels in the input does not fit that of the denoiser
        with torch.no_grad():
            return torch2xp(self.dncnn(xp2torch(input_image)))
