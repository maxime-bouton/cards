import numpy as np
import torch

from mcmc.backend import xp
from mcmc.denoisers.base_denoiser import BaseDenoiser
from mcmc.denoisers.denoiser_loader import load_pretrained_dncnn
from mcmc.utils.utils import torch2xp, xp2torch


class SerialDnCNN(BaseDenoiser):
    def __init__(self, image_size: np.ndarray):
        """
        Serial DnCNN.

        Parameters
        ----------
        image_size: np.ndarray
            The input shape
        """

        self.dncnn = load_pretrained_dncnn(image_size[-3])

    def __call__(self, input_image: xp.ndarray, sigma: float) -> xp.ndarray:
        with torch.no_grad():
            return torch2xp(self.dncnn(xp2torch(input_image)))
