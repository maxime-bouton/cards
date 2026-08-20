r"""Distributed denoiser class for the DnCNN network :cite:p:`Zhang2017`."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from pathlib import Path

import torch
from mpi4py import MPI

import cards.backend as xp
from cards.denoisers.base_denoiser import BaseDistributedDenoiser
from cards.denoisers.denoiser_loader import load_pretrained_dncnn
from cards.operators.distributed_torch_convolution import DistributedTorchConvolution

# FIXME: rename all internal convolution operators to make it clear they are private


class DistributedDnCNN(BaseDistributedDenoiser):
    r"""Distributed implementation of the DnCNN :cite:p:`Zhang2017` network.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Underlying MPI communicator.
    grid_shape : tuple[int, ...]
        Number of MPI workers along each of axis of the communicator grid.
    image_shape: tuple[int, ...]
        Input image shape.
    weights_path : str, optional
        Path to the folder containing the pre-trained denoiser weights.

    Attributes
    ----------
    core_mpi_conv : DistributedTorchConvolution
        Distributed convolution operator implemented in torch, corresponding to the inner layers of the network.
    edge_mpi_conv : DistributedTorchConvolution
        Distributed convolution operator implemented in torch, corresponding to the first and the last layer of the network.
    dncnn : DnCNN
        Local DnCNN denoiser.
    """

    def __init__(
        self,
        comm: MPI.Comm,
        grid_shape: tuple[int, ...],
        image_shape: tuple[int, ...],
        weights_path=Path(__file__).parents[3] / "data/weights/dncnn",
    ):
        super(BaseDistributedDenoiser, self).__init__(weights_path)
        if len(image_shape) < 3:
            # NOTE: accommodate gray scale images (implicitly, number of channes is 1)
            n_channels = 1
            core_shape = [1, *image_shape]
        else:
            n_channels = image_shape[-3]
            core_shape = list(image_shape)

        self.dncnn = load_pretrained_dncnn(
            n_channels,
            weights_path=self.weights_path,
        )

        self.edge_mpi_conv = DistributedTorchConvolution(
            image_shape,
            self.dncnn.model[0].kernel_size,
            self.dncnn.model[0].padding,
            comm,
            grid_shape,
            Cout=self.dncnn.model[0].out_channels,
            backward=False,
        )

        core_shape[-3] = self.dncnn.model[0].out_channels
        core_shape = tuple(core_shape)

        tile_range = (
            self.edge_mpi_conv.adjoint_communicator.cartslicer.tile_range.copy()
        )
        tile_range[-3] = [0, self.dncnn.model[0].out_channels - 1]

        self.core_mpi_conv = DistributedTorchConvolution(
            core_shape,
            self.dncnn.model[0].kernel_size,
            self.dncnn.model[0].padding,
            comm,
            grid_shape,
            backward=True,
            tile_range=tile_range,
        )

    def _apply_layer(
        self,
        tile_u: xp.ndarray,
        conv_forward: torch.nn.Conv2d,
        conv_adjoint: torch.nn.ConvTranspose2d,  # torch.nn.Conv2d,
    ) -> xp.ndarray:
        tile_u = self.core_mpi_conv.forward(tile_u, op=conv_forward).clip(min=0)
        return self.core_mpi_conv.adjoint(tile_u, adjoint_op=conv_adjoint).clip(min=0)

    def __call__(
        self, input_image: xp.ndarray, sigma: float, torch_dtype=None, xp_dtype=None
    ) -> xp.ndarray:
        r"""Apply the distributed denoiser.

        Parameters
        ----------
        input_image: xp.ndarray
            Input image tile.
        sigma: float
            Denoiser parameter (noise standard deviation).

        Returns
        -------
        xp.ndarray
            Denoised image tile.
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
        r"""Apply the denoiser without communication on the first layer.

        This method can be used to avoid memory duplication when the input
        buffer is common to different operators, thereby reducing the overall
        memory footprint.

        Parameters
        ----------
        input_image : xp.ndarray
            Input image facet (i.e., image buffer including ghost-cell) to denoise.
        sigma: float
            Denoiser parameter (noise standard deviation).

        Returns
        -------
        xp.ndarray
            Denoised image.
        """
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

    @property
    def get_recv_size(self) -> xp.ndarray:
        return self.edge_mpi_conv.direct_communicator.cartslicer.recv_size

    @property
    def get_send_size(self) -> xp.ndarray:
        return self.edge_mpi_conv.direct_communicator.cartslicer.send_size

    @property
    def global_to_tile_slice(self):
        return self.edge_mpi_conv.direct_communicator.cartslicer.slice_global_buffer_to_tile
