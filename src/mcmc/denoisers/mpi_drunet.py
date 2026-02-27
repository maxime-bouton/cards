import numpy as np
import torch
from mpi4py import MPI

from mcmc.backend import xp
from mcmc.denoisers.base_denoiser import BaseDistributedDenoiser
from mcmc.denoisers.denoiser_loader import load_pretrained_drunet
from mcmc.operators.mpi_torch_convolution import MpiTorchConvolution
from mcmc.utils.utils import torch2xp, xp2torch


class MpiDRUNet(BaseDistributedDenoiser):
    def __init__(self, comm: MPI.Comm, grid_size: np.ndarray, image_size: np.ndarray):
        """
        Distributed DRUNet.

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
        """

        self.drunet = load_pretrained_drunet(image_size[-3])

        # full size concatenated with noise map channel
        size_in = image_size.copy()
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

    def __call__(self, input_image: xp.ndarray, sigma: float) -> xp.ndarray:
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
