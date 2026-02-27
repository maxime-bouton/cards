import numpy as np
import torch
from mpi4py import MPI

from mcmc.backend import xp
from mcmc.denoisers.base_denoiser import BaseDistributedDenoiser
from mcmc.denoisers.denoiser_loader import load_pretrained_dncnn
from mcmc.operators.mpi_torch_convolution import MpiTorchConvolution


class MpiDnCNN(BaseDistributedDenoiser):
    def __init__(self, comm: MPI.Comm, grid_size: np.ndarray, image_size: np.ndarray):
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
        """

        self.dncnn = load_pretrained_dncnn(image_size[-3])

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

    @property
    def global_to_tile_slice(self):
        return self.edge_mpi_conv.direct_communicator.cartslicer._get_slice_global_buffer_to_tile()

    @property
    def state_shape(self) -> np.ndarray:
        return self.edge_mpi_conv.direct_communicator.cartslicer.tile_size
