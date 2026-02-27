from typing import Sequence, cast

import torch
from torch.nn import Conv2d, ConvTranspose2d, Module, ModuleList

from mcmc.utils.utils import power_method


class DFBLayer(Module):
    """
    A single Dual Forward-Backward layer optimized for minimal memory footprint.

    Parameters
    ----------
    Cin : int
        Number of input channels.
    Cout : int
        Number of output channels.
    ksize : int, optional
        Kernel size (default is 3).
    pad : int, optional
        Padding (default is 1).
    """

    def __init__(self, Cin: int, Cout: int, ksize: int = 3, pad: int = 1):
        super().__init__()
        self.Dk = Conv2d(Cin, Cout, ksize, padding=pad, bias=False)
        self.Dk_T = ConvTranspose2d(Cout, Cin, ksize, padding=pad, bias=False)
        self.Dk_T.weight = self.Dk.weight
        self.tau_k: float | None = None

    def forward(
        self,
        u: torch.Tensor,
        x_ref: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        if self.training:
            self.update_lip((1, *x_ref.shape[-3:]))

            tmp = (x_ref - self.Dk_T(u)).clamp(0, 1)
            return (u + self.tau_k * self.Dk(tmp)).clamp(-nu, nu)
        else:
            if self.tau_k is None:
                self.update_lip((1, *x_ref.shape[-3:]))

            tmp = x_ref.clone().sub_(self.Dk_T(u)).clamp_(0, 1)
            return u.add_(self.Dk(tmp), alpha=self.tau_k).clamp_(-nu, nu)

    @torch.no_grad()
    def update_lip(self, im_size: Sequence[int], rng=None):
        self.norm_Dk2 = power_method(self.Dk, self.Dk_T, im_size, rng=rng)
        self.tau_k = 1.99 / self.norm_Dk2


class DDFB(Module):
    """
    Deep Dual Forward-Backward (DDFB) model.

    Parameters
    ----------
    C : int, optional
        Number of input channels (default is 3).
    n_layers : int, optional
        Number of layers (default is 5).
    n_features : int, optional
        Number of features (default is 64).
    """

    def __init__(self, C: int = 3, n_layers: int = 5, n_features: int = 64):
        super().__init__()
        ksize, pad = 3, 1
        self.D0 = Conv2d(C, n_features, ksize, padding=pad, bias=False)
        self.D0_T = ConvTranspose2d(n_features, C, ksize, padding=pad, bias=False)
        self.D0_T.weight = self.D0.weight
        self.layers = ModuleList([DFBLayer(C, n_features) for _ in range(n_layers - 1)])

    def forward(
        self,
        x: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        u = self.D0(x)
        for block in self.layers:
            u = block(u, x, nu)
        return (x - self.D0_T(u)).clamp(0, 1)

    @torch.no_grad()
    def update_lip(self, im_size: Sequence[int], rng=None):
        self._lip2_layer0 = power_method(self.D0, self.D0_T, im_size, rng=rng)
        for block in self.layers:
            cast(DFBLayer, block).update_lip(im_size, rng)

    def get_lip2_values(self) -> list[float]:
        return [self._lip2_layer0] + [cast(DFBLayer, b).norm_Dk2 for b in self.layers]
