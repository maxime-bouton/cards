import numpy as np
import torch
from mpi4py import MPI

from mcmc.backend import xp
from mcmc.denoisers.base_denoiser import BaseDistributedDenoiser
from mcmc.denoisers.ddfb.network_ddfb import DFBLayer
from mcmc.denoisers.denoiser_loader import load_pretrained_ddfb
from mcmc.operators.mpi_torch_convolution import MpiTorchConvolution


class MpiDDFB(BaseDistributedDenoiser):
    def __init__(
        self,
        comm: MPI.Comm,
        grid_size: np.ndarray,
        image_size: np.ndarray,
        n_layers: int,
        n_features: int,
    ):
        """
        Distributed DDFB.

        Parameters
        ----------
        image_size: np.ndarray
            The input shape
        n_layers: int
            The number of DFBLayers
        n_features: int
            The number of channels in the dual space (i.e. number of channels for `u`)
        comm: BaseCartesianCommunicator
            The Cartesian communicator
        timer: TimerRegistry, optional
            The timer registry
        logger: logging.Logger, optional
            If unspecified, no logging will be displayed (default is None).
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

        self.mpi_conv = MpiTorchConvolution(
            image_size,
            self.ddfb.D0.kernel_size,
            self.ddfb.D0.padding,  # type: ignore
            comm,
            grid_size,
            self.ddfb.D0.out_channels,
        )

    def _apply_layer(
        self,
        tile_u: xp.ndarray,
        tile_x_ref: xp.ndarray,
        nu: float,
        layer: DFBLayer,
    ) -> xp.ndarray:
        Dk_T_u = self.mpi_conv.adjoint(tile_u, layer.Dk_T)
        Dk_tmp = self.mpi_conv.forward((tile_x_ref - Dk_T_u).clip(0, 1), layer.Dk)
        return (tile_u + layer.tau_k * Dk_tmp).clip(-nu, nu)

    def __call__(self, input_image: xp.ndarray, sigma: float) -> xp.ndarray:
        """
        Apply the distributed DDFB.

        Parameters
        ----------
        input_image: xp.ndarray
            The input x facet
        sigma: float
            The regularization parameter

        Returns
        -------
        xp.ndarray
            The denoised x tile
        """
        tile_u = self.mpi_conv.forward(input_image, self.ddfb.D0)
        for layer in self.ddfb.layers:
            tile_u = self._apply_layer(
                tile_u,
                input_image,
                sigma,
                layer,  # type: ignore
            )
        D0_T_u = self.mpi_conv.adjoint(tile_u, self.ddfb.D0_T)
        return (input_image - D0_T_u).clip(0, 1)

    @property
    def global_to_tile_slice(self):
        return self.mpi_conv.direct_communicator.cartslicer._get_slice_global_buffer_to_tile()

    @property
    def state_shape(self) -> np.ndarray:
        return self.mpi_conv.direct_communicator.cartslicer.tile_size
