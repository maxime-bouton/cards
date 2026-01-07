r"""Distributed denoiser class for the Deep Dual Forward-Backward (DDFB) network :cite:`Repetti2022eusipco`."""

from pathlib import Path

import numpy as np
import torch
from mpi4py import MPI

from cards.backend import xp
from cards.denoisers.base_denoiser import BaseDistributedDenoiser
from cards.denoisers.ddfb.network_ddfb import DFBLayer
from cards.denoisers.denoiser_loader import load_pretrained_ddfb
from cards.operators.mpi_torch_convolution import MpiTorchConvolution

# FIXME: investigate older shared_comm branch (some input not used)
# FIXME: shared_input_buffer not used, revise implementation


class MpiDDFB(BaseDistributedDenoiser):
    def __init__(
        self,
        comm: MPI.Comm,
        grid_size: np.ndarray,
        image_size: np.ndarray,
        n_layers: int,
        n_features: int,
        weights_path=Path(__file__).parents[3] / "data/weights/ddfb",
        # use_shared_input_buffer: bool = False,
    ):
        r"""
        Distributed DDFB.

        Parameters
        ----------
        image_size: np.ndarray
            The input shape
        n_layers: int
            The number of DFBLayers
        n_features: int
            The number of channels in the dual space of the convolution operators.
        comm: BaseCartesianCommunicator
            The Cartesian communicator
        timer: TimerRegistry, optional
            The timer registry
        logger: logging.Logger, optional
            If unspecified, no logging will be displayed (default is None).
        weights_path : str, optional
            The path to the pre-trained weights folder.
        """
        super(BaseDistributedDenoiser, self).__init__(weights_path)
        if image_size.size < 3:
            # NOTE: accommodate gray scale images (implicitly, number of channes is 1)
            n_channels = 1
            im_size = xp.concatenate((xp.ones(1, dtype=image_size.dtype), image_size))
        else:
            n_channels = image_size[-3]
            im_size = image_size.copy()
        self.n_features = n_features
        self.n_layers = n_layers

        self.ddfb = load_pretrained_ddfb(
            n_channels,
            n_layers=n_layers,
            n_features=n_features,
            weights_path=self.weights_path,
        )

        rng = torch.Generator(next(self.ddfb.parameters()).device).manual_seed(42)
        self.ddfb.update_lip(tuple(im_size[-3:]), rng=rng)

        self.mpi_conv = MpiTorchConvolution(
            im_size,
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
            The input x tile. Tile size.
        sigma: float
            The regularization parameter

        Returns
        -------
        xp.ndarray
            The denoised x tile
        """
        tile_u = self.mpi_conv.forward(input_image.clip(0, 1), self.ddfb.D0)
        for layer in self.ddfb.layers:
            tile_u = self._apply_layer(
                tile_u,
                input_image,
                sigma,
                layer,  # type: ignore
            )
        D0_T_u = self.mpi_conv.adjoint(tile_u, self.ddfb.D0_T)
        return (input_image - D0_T_u).clip(0, 1)

    def forward_no_comm(self, input_image: xp.ndarray, sigma: float) -> xp.ndarray:
        """Apply the DDFB denoiser without communication on the first layer.

        This method can be used to avoid memory duplication when the input
        buffer is common to different operators, thereby reducing the overall
        memory footprint.

        Parameters
        ----------
        input_image : xp.ndarray
            Image facet (i.e., including ghost-cell) to denoise.
        sigma : float
            Standard deviation of the Gaussian noise.

        Returns
        -------
        xp.ndarray
            Denoised image facet.
        """
        assert isinstance(sigma, float)  #! fail on np.float or torch.float?
        tile_u = self.mpi_conv.forward_no_comm(input_image.clip(0, 1), self.ddfb.D0)
        for layer in self.ddfb.layers:
            tile_u = self._apply_layer(
                tile_u,
                input_image[
                    self.mpi_conv.direct_communicator.cartslicer.slice_facet_to_tile
                ],
                sigma,
                layer,  # type: ignore
            )
        D0_T_u = self.mpi_conv.adjoint(tile_u, self.ddfb.D0_T)
        return (
            input_image[
                self.mpi_conv.direct_communicator.cartslicer.slice_facet_to_tile
            ]
            - D0_T_u
        ).clip(0, 1)

    @property
    def get_recv_size(self) -> np.ndarray:
        return self.mpi_conv.direct_communicator.cartslicer.recv_size

    @property
    def get_send_size(self) -> np.ndarray:
        return self.mpi_conv.direct_communicator.cartslicer.send_size

    @property
    def global_to_tile_slice(self):
        return self.mpi_conv.direct_communicator.cartslicer._get_slice_global_buffer_to_tile()
