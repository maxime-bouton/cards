r"""Distributed denoiser class for the DRUNet network :cite:p:`Zhang2021`."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from pathlib import Path

import numpy as np
import torch
from mpi4py import MPI

import cards.backend as xp
from cards.denoisers.base_denoiser import BaseDistributedDenoiser
from cards.denoisers.denoiser_loader import load_pretrained_drunet
from cards.operators.distributed_torch_convolution import DistributedTorchConvolution
from cards.utils.utils import torch2xp, xp2torch

# TODO: add method equivalent to forward_no_comm in DDFB and DnCNN
# FIXME: rename all internal convolution operators to make it clear they are private


class DistributedDRUNet(BaseDistributedDenoiser):
    r"""Distributed implementation of the DRUNet :cite:p:`Zhang2021` network.

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
    conv1, conv2, conv3, conv4 : DistributedTorchConvolution
        Internal convolution operators used to defined the layers of the network (after different up-/down-sampling levels).
    head_conv : DistributedTorchConvolution
        Convolution operator in the first layer of the network.
    tail_conv : DistributedTorchConvolution
        Convolution operator in the last layer of the network.
    drunet : DRUNet
        Internal DRUNet denoiser.

    Methods
    -------
    __call__()
        Apply the distributed denoiser to an input image.

    Warning
    -------
    The current distributed implementation only works for images whose
    spatial dimensions are a multiple of 8. More precisely, the spatial
    shape of the local tiles handled by each worker needs to be a
    multiple of 8 to avoid additional communications for the up- and
    down-sampling operators.
    """

    def __init__(
        self,
        comm: MPI.Comm,
        grid_shape: tuple[int, ...],
        image_shape: tuple[int, ...],
        weights_path=Path(__file__).parents[3] / "data/weights/drunet",
    ):
        super(BaseDistributedDenoiser, self).__init__(weights_path)
        if len(image_shape) < 3:
            # NOTE: accommodate gray scale images (implicitly, number of channes is 1)
            n_channels = 1
            size_in = [1, *image_shape]
        else:
            n_channels = image_shape[-3]
            size_in = list(image_shape)

        self.drunet = load_pretrained_drunet(
            n_channels,
            weights_path=self.weights_path,
        )

        # full size concatenated with noise map channel
        # size_in = image_size.copy()
        size_in[-3] += 1
        size_in = tuple(size_in)

        # full size in dual space
        size1 = np.asarray(size_in)
        size1[-3] = self.drunet.m_head.out_channels

        # channel doubled and spatial dimensions halved for each downsampling level
        size2 = size1.copy()
        size2[-3] = size1[-3] * 2
        size2[-2:] = size1[-2:] // 2

        size3 = size2.copy()
        size3[-3] = size2[-3] * 2
        size3[-2:] = size2[-2:] // 2

        size4 = size3.copy()
        size4[-3] = size3[-3] * 2
        size4[-2:] = size3[-2:] // 2

        self.conv1 = DistributedTorchConvolution(
            (*size1,),
            self.drunet.m_down1[0].res[0].kernel_size,
            self.drunet.m_down1[0].res[0].padding,
            comm,
            grid_shape,
        )

        self.conv2 = DistributedTorchConvolution(
            (*size2,),
            self.drunet.m_down2[0].res[0].kernel_size,
            self.drunet.m_down2[0].res[0].padding,
            comm,
            grid_shape,
        )

        self.conv3 = DistributedTorchConvolution(
            (*size3,),
            self.drunet.m_down3[0].res[0].kernel_size,
            self.drunet.m_down3[0].res[0].padding,
            comm,
            grid_shape,
        )

        self.conv4 = DistributedTorchConvolution(
            (*size4,),
            self.drunet.m_body[0].res[0].kernel_size,
            self.drunet.m_body[0].res[0].padding,
            comm,
            grid_shape,
        )

        tile_range = self.conv1.adjoint_communicator.cartslicer.tile_range.copy()
        tile_range[-3] = [0, self.drunet.m_head.in_channels - 1]

        self.head_conv = DistributedTorchConvolution(
            size_in,
            self.drunet.m_head.kernel_size,
            self.drunet.m_head.padding,
            comm,
            grid_shape,
            Cout=self.drunet.m_head.out_channels,
            backward=True,
            tile_range=tile_range,
        )

        self.tail_conv = DistributedTorchConvolution(
            (*size1,),
            self.drunet.m_tail.kernel_size,
            self.drunet.m_tail.padding,
            comm,
            grid_shape,
            Cout=image_shape[-3],
            backward=False,
        )

    def _apply_res_layer(
        self,
        tile_u: xp.ndarray,
        mpi_conv: DistributedTorchConvolution,
        conv_forward: torch.nn.Conv2d,
        conv_adjoint: torch.nn.ConvTranspose2d,
    ) -> xp.ndarray:
        tmp = mpi_conv.forward(tile_u, op=conv_forward)
        tmp.clip(min=0, out=tmp)
        result = mpi_conv.adjoint(tmp, adjoint_op=conv_adjoint)
        result += tile_u
        return result

    def _apply_down_level(
        self,
        tile_u: xp.ndarray,
        level: torch.nn.Module,
        conv: DistributedTorchConvolution,
    ) -> xp.ndarray:
        for res in level[:-1]:
            tile_u = self._apply_res_layer(tile_u, conv, res.res[0], res.res[2])
        return torch2xp(level[-1](xp2torch(tile_u)))

    def _apply_up_level(
        self,
        tile_u: xp.ndarray,
        level: torch.nn.Module,
        conv: DistributedTorchConvolution,
    ) -> xp.ndarray:
        tile_u = torch2xp(level[0](xp2torch(tile_u)))
        for res in level[1:]:
            tile_u = self._apply_res_layer(tile_u, conv, res.res[0], res.res[2])
        return tile_u

    def __call__(
        self, input_image: xp.ndarray, sigma: float, torch_dtype=None, xp_dtype=None
    ) -> xp.ndarray:
        _, h, w = input_image.shape
        noise_map = xp.full((1, h, w), sigma)
        tile_x0 = xp.concatenate((input_image, noise_map), axis=0)
        tile_x1 = self.head_conv.forward(tile_x0, self.drunet.m_head)
        tile_x2 = self._apply_down_level(tile_x1, self.drunet.m_down1, self.conv1)
        tile_x3 = self._apply_down_level(tile_x2, self.drunet.m_down2, self.conv2)
        tile_x4 = self._apply_down_level(tile_x3, self.drunet.m_down3, self.conv3)

        tile_x = tile_x4.copy()
        for res in self.drunet.m_body:
            tile_x = self._apply_res_layer(tile_x, self.conv4, res.res[0], res.res[2])

        tile_x = self._apply_up_level(tile_x + tile_x4, self.drunet.m_up3, self.conv3)
        tile_x = self._apply_up_level(tile_x + tile_x3, self.drunet.m_up2, self.conv2)
        tile_x = self._apply_up_level(tile_x + tile_x2, self.drunet.m_up1, self.conv1)
        return self.tail_conv.forward(tile_x + tile_x1, self.drunet.m_tail)

    @property
    def global_to_tile_slice(self):
        return self.tail_conv.adjoint_communicator.cartslicer._get_slice_global_buffer_to_tile()

    @property
    def state_shape(self) -> np.ndarray:
        r"""Returns the shape of the local output image tile handled by the current process."""
        return self.tail_conv.adjoint_communicator.cartslicer.tile_size

    @property
    def tile_range(self) -> np.ndarray | None:
        r"""Returns the start and end index of the output image tile handled by the current worker."""
        return self.tail_conv.adjoint_communicator.cartslicer.tile_range

    # FIXME: check implementation of the two methods below (if ever useful)
    @property
    def get_recv_size(self) -> xp.ndarray:
        return self.tail_conv.adjoint_communicator.cartslicer.recv_size

    @property
    def get_send_size(self) -> xp.ndarray:
        return self.tail_conv.adjoint_communicator.cartslicer.send_size
