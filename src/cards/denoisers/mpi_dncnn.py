r"""Distributed denoiser class for the DnCNN network :cite:`Zhang2017`."""

from pathlib import Path

import numpy as np
import torch
from mpi4py import MPI

from cards.backend import xp
from cards.denoisers.base_denoiser import BaseDistributedDenoiser
from cards.denoisers.denoiser_loader import load_pretrained_dncnn
from cards.operators.mpi_torch_convolution import MpiTorchConvolution


class MpiDnCNN(BaseDistributedDenoiser):
    def __init__(
        self,
        comm: MPI.Comm,
        grid_size: np.ndarray,
        image_size: np.ndarray,
        weights_path=Path(__file__).parents[3] / "data/weights/dncnn",
    ):
        """
        Distributed DnCNN.

        Parameters
        ----------
        image_size: np.ndarray
            The input shape
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

        self.dncnn = load_pretrained_dncnn(
            image_size[-3],
            weights_path=self.weights_path,
        )

        self.edge_mpi_conv = MpiTorchConvolution(
            image_size,
            self.dncnn.model[0].kernel_size,
            self.dncnn.model[0].padding,  # type: ignore
            comm,
            grid_size,
            Cout=self.dncnn.model[0].out_channels,
            backward=False,
        )

        core_size = image_size.copy()
        core_size[-3] = self.dncnn.model[0].out_channels

        tile_range = (
            self.edge_mpi_conv.adjoint_communicator.cartslicer.tile_range.copy()
        )
        tile_range[-3] = [0, self.dncnn.model[0].out_channels - 1]

        self.core_mpi_conv = MpiTorchConvolution(
            core_size,
            self.dncnn.model[0].kernel_size,
            self.dncnn.model[0].padding,  # type: ignore
            comm,
            grid_size,
            backward=True,
            tile_range=tile_range,
        )

    def _apply_layer(
        self,
        tile_u: xp.ndarray,
        conv_forward: torch.nn.Conv2d,
        conv_adjoint: torch.nn.Conv2d,
    ) -> xp.ndarray:
        tile_u = self.core_mpi_conv.forward(tile_u, conv_forward).clip(min=0)
        return self.core_mpi_conv.adjoint(tile_u, conv_adjoint).clip(min=0)

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
        tile_u = self.edge_mpi_conv.forward(input_image, self.dncnn.model[0]).clip(
            min=0
        )
        for i in range(len(self.dncnn.model[2:-2:4])):
            conv_forward = self.dncnn.model[2 + 4 * i]
            # NOTE: This is not the actual ADJOINT convolution: convenience for communications handling
            conv_adjoint = self.dncnn.model[2 + 4 * i + 2]
            tile_u = self._apply_layer(tile_u, conv_forward, conv_adjoint)

        return input_image - self.edge_mpi_conv.adjoint(tile_u, self.dncnn.model[-1])

    def forward_no_comm(self, input_image: xp.ndarray, sigma: float) -> xp.ndarray:
        """forward_no_comm Apply the denoiser without communication on the first layer. Should be used with a shared buffer.

        Parameters
        ----------
        input_image : xp.ndarray
            Image to denoise. Facet size.
        sigma : float
            Standard deviation of the gaussian noise.

        Returns
        -------
        xp.ndarray
            Denoised image.
        """
        assert isinstance(sigma, float)  #! fail on np.float or torch.float?
        tile_u = self.edge_mpi_conv.forward_no_comm(
            input_image, self.dncnn.model[0]
        ).clip(min=0)
        for i in range(len(self.dncnn.model[2:-2:4])):
            conv_forward = self.dncnn.model[2 + 4 * i]
            # NOTE: This is not the actual ADJOINT convolution: convenience for communications handling
            conv_adjoint = self.dncnn.model[2 + 4 * i + 2]
            tile_u = self._apply_layer(tile_u, conv_forward, conv_adjoint)

        return input_image[
            self.edge_mpi_conv.direct_communicator.cartslicer.slice_facet_to_tile
        ] - self.edge_mpi_conv.adjoint(tile_u, self.dncnn.model[-1])

    def get_recv_size(self) -> np.ndarray:
        return self.edge_mpi_conv.direct_communicator.cartslicer.recv_size

    def get_send_size(self) -> np.ndarray:
        return self.edge_mpi_conv.direct_communicator.cartslicer.send_size

    @property
    def global_to_tile_slice(self):
        return self.edge_mpi_conv.direct_communicator.cartslicer._get_slice_global_buffer_to_tile()
