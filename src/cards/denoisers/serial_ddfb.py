r"""Serial denoiser class for the Deep Dual Forward-Backward (DDFB) network :cite:`Repetti2022eusipco`."""

from pathlib import Path

import numpy as np
import torch

from cards.backend import xp
from cards.denoisers.base_denoiser import BaseDenoiser
from cards.denoisers.denoiser_loader import load_pretrained_ddfb
from cards.utils.utils import torch2xp, xp2torch


class SerialDDFB(BaseDenoiser):
    def __init__(
        self,
        image_size: np.ndarray,
        n_layers: int,
        n_features: int,
        weights_path=Path(__file__).parents[3] / "data/weights/ddfb",
    ):
        """
        Serial DDFB.

        Parameters
        ----------
        image_size: np.ndarray
            The input shape
        n_layers: int
            The number of DFBLayers
        n_features: int
            The number of channels in the dual space (i.e. number of channels for `u`)
        weights_path : str, optional
            The path to the pre-trained weights folder.
        """
        super(SerialDDFB, self).__init__(weights_path)

        self.ddfb = load_pretrained_ddfb(
            image_size[-3],
            n_layers=n_layers,
            n_features=n_features,
            weights_path=self.weights_path,
        )

        rng = torch.Generator(next(self.ddfb.parameters()).device).manual_seed(42)
        self.ddfb.update_lip(tuple(image_size[-3:]), rng=rng)

    def __call__(self, input_image: xp.ndarray, sigma: float) -> xp.ndarray:
        with torch.no_grad():
            return torch2xp(self.ddfb(xp2torch(input_image), sigma))
