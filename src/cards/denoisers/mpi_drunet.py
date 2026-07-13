r"""Distributed denoiser class for the DRUNet network :cite:p:`Zhang2021`."""

from pathlib import Path

import numpy as np
import torch
from mpi4py import MPI

import cards.backend as xp
from cards.denoisers.base_denoiser import BaseDistributedDenoiser
from cards.denoisers.denoiser_loader import load_pretrained_drunet
from cards.operators.mpi_torch_convolution import MpiTorchConvolution
from cards.utils.utils import torch2xp, xp2torch

# TODO: add method equivalent to forward_no_comm in DDFB and DnCNN
# FIXME: rename all internal convolution operators to make it clear they are private


class DistributedDRUNet(BaseDistributedDenoiser):
    r"""Distributed implementation of the DRUNet :cite:p:`Zhang2021` network.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Underlying MPI communicator.
    grid_size : xp.ndarray[int]
        Number of workers along each of the ``d`` dimensions of the
        communicator grid.
    image_size: xp.ndarray
        Input image shape.
    weights_path : str, optional
        Path to the folder containing the pre-trained denoiser weights.

    Attributes
    ----------
    conv1, conv2, conv3, conv4 : MpiTorchConvolution
        Internal convolution operators used to defined the layers of the network (after different up-/down-sampling levels).
    head_conv : MpiTorchConvolution
        Convolution operator in the first layer of the network.
    tail_conv : MpiTorchConvolution
        Convolution operator in the last layer of the network.
    drunet : DRUNet
        Internal DRUNet denoiser.

    Methods
    -------
    state_shape()
        Returns the shape of the local output image tile handled by the current process.
    tile_range()
        Returns the start and end index of the output image tile handled by the current worker.

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
        grid_size: np.ndarray,
        image_size: np.ndarray,
        weights_path=Path(__file__).parents[3] / "data/weights/drunet",
    ):
        super(BaseDistributedDenoiser, self).__init__(weights_path)
        if image_size.size < 3:
            # NOTE: accommodate gray scale images (implicitly, number of channes is 1)
            n_channels = 1
            size_in = xp.concatenate((xp.ones(1, dtype=image_size.dtype), image_size))
        else:
            n_channels = image_size[-3]
            size_in = image_size.copy()

        self.drunet = load_pretrained_drunet(
            n_channels,
            weights_path=self.weights_path,
        )

        # full size concatenated with noise map channel
        # size_in = image_size.copy()
        size_in[-3] += 1

        # full size in dual space
        size1 = size_in.copy()
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

        self.conv1 = MpiTorchConvolution(
            size1,
            self.drunet.m_down1[0].res[0].kernel_size,  # type: ignore
            self.drunet.m_down1[0].res[0].padding,  # type: ignore
            comm,
            grid_size,
        )

        self.conv2 = MpiTorchConvolution(
            size2,
            self.drunet.m_down2[0].res[0].kernel_size,  # type: ignore
            self.drunet.m_down2[0].res[0].padding,  # type: ignore
            comm,
            grid_size,
        )

        self.conv3 = MpiTorchConvolution(
            size3,
            self.drunet.m_down3[0].res[0].kernel_size,  # type: ignore
            self.drunet.m_down3[0].res[0].padding,  # type: ignore
            comm,
            grid_size,
        )

        self.conv4 = MpiTorchConvolution(
            size4,
            self.drunet.m_body[0].res[0].kernel_size,  # type: ignore
            self.drunet.m_body[0].res[0].padding,  # type: ignore
            comm,
            grid_size,
        )

        tile_range = self.conv1.adjoint_communicator.cartslicer.tile_range.copy()
        tile_range[-3] = [0, self.drunet.m_head.in_channels - 1]

        self.head_conv = MpiTorchConvolution(
            size_in,
            self.drunet.m_head.kernel_size,  # type: ignore
            self.drunet.m_head.padding,  # type: ignore
            comm,
            grid_size,
            Cout=self.drunet.m_head.out_channels,  # type: ignore
            backward=True,
            tile_range=tile_range,
        )

        self.tail_conv = MpiTorchConvolution(
            size1,
            self.drunet.m_tail.kernel_size,  # type: ignore
            self.drunet.m_tail.padding,  # type: ignore
            comm,
            grid_size,
            Cout=image_size[-3],
            backward=False,
        )

    def _apply_res_layer(
        self,
        tile_u: xp.ndarray,
        mpi_conv: MpiTorchConvolution,
        conv_forward: torch.nn.Conv2d,
        conv_adjoint: torch.nn.Conv2d,
    ) -> xp.ndarray:
        tmp = mpi_conv.forward(tile_u, conv_forward)
        tmp.clip(min=0, out=tmp)
        result = mpi_conv.adjoint(tmp, conv_adjoint)
        result += tile_u
        return result

    def _apply_down_level(
        self,
        tile_u: xp.ndarray,
        level: torch.nn.Module,
        conv: MpiTorchConvolution,
    ) -> xp.ndarray:
        for res in level[:-1]:
            tile_u = self._apply_res_layer(tile_u, conv, res.res[0], res.res[2])
        return torch2xp(level[-1](xp2torch(tile_u)))

    def _apply_up_level(
        self,
        tile_u: xp.ndarray,
        level: torch.nn.Module,
        conv: MpiTorchConvolution,
    ) -> xp.ndarray:
        tile_u = torch2xp(level[0](xp2torch(tile_u)))
        for res in level[1:]:
            tile_u = self._apply_res_layer(tile_u, conv, res.res[0], res.res[2])
        return tile_u

    def __call__(
        self, input_image: xp.ndarray, sigma: float, torch_dtype=None, cp_dtype=None
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
        return self.tail_conv.adjoint_communicator.cartslicer.tile_size

    @property
    def tile_range(self) -> np.ndarray | None:
        return self.tail_conv.adjoint_communicator.cartslicer.tile_range
