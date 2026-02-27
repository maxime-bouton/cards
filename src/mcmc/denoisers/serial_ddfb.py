import numpy as np
import torch

from mcmc.backend import xp
from mcmc.denoisers.base_denoiser import BaseDenoiser
from mcmc.denoisers.denoiser_loader import load_pretrained_ddfb
from mcmc.utils.utils import torch2xp, xp2torch


class SerialDDFB(BaseDenoiser):
    def __init__(
        self,
        image_size: np.ndarray,
        n_layers: int,
        n_features: int,
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
        """
        self.n_features = n_features
        self.n_layers = n_layers

        self.ddfb = load_pretrained_ddfb(
            image_size[-3],
            n_layers=n_layers,
            n_features=n_features,
        )

        rng = torch.Generator(next(self.ddfb.parameters()).device).manual_seed(42)
        self.ddfb.update_lip(tuple(image_size[-3:]), rng=rng)

    def __call__(self, input_image: xp.ndarray, sigma: float) -> xp.ndarray:
        with torch.no_grad():
            return torch2xp(self.ddfb(xp2torch(input_image), sigma))
