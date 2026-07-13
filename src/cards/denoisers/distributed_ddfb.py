r"""Distributed denoiser class for the Deep Dual Forward-Backward (DDFB) network :cite:p:`Repetti2022eusipco`."""

from pathlib import Path

import torch
from mpi4py import MPI

import cards.backend as xp
from cards.denoisers.base_denoiser import BaseDistributedDenoiser
from cards.denoisers.ddfb.network_ddfb import DFBLayer
from cards.denoisers.denoiser_loader import load_pretrained_ddfb
from cards.operators.mpi_torch_convolution import MpiTorchConvolution

# TODO: move forward_no_comm method to the base class (variant of __call__() which does not trigger communications for the input stage of the network, which can possibly be made in common with other operators)
# FIXME: rename all internal convolution operators to make it clear they are private


class DistributedDDFB(BaseDistributedDenoiser):
    r"""Distributed implementation for the Deep Dual Forward-Backward (DDFB)
    denoiser :cite:p:`Repetti2022eusipco`.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Underlying MPI communicator.
    grid_size : xp.ndarray[int]
        Number of workers along each of the ``d`` dimensions of the
        communicator grid.
    image_size: xp.ndarray
        Input image shape.
    n_layers: int
        Number of DDFB layers.
    n_features: int
        Number of channels in the dual space of the convolution operators.
    weights_path : str, optional
        Path to the folder containing the pre-trained denoiser weights.

    Attributes
    ----------
    n_layers: int
        Number of DDFB layers.
    n_features: int
        Number of channels in the dual space of the convolution operators.
    mpi_conv: MpiTorchConvolution
        Internal distributed convolution operator implemented in torch.
    ddfb : DDFB
        Local DDFB denoiser.

    Methods
    -------
    forward_no_comm()
        Apply the DDFB denoiser without communication for the first layer.
    """

    def __init__(
        self,
        comm: MPI.Comm,
        grid_size: xp.ndarray,
        image_size: xp.ndarray,
        n_layers: int,
        n_features: int,
        weights_path=Path(__file__).parents[3] / "data/weights/ddfb",
    ):
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

    def __call__(
        self, input_image: xp.ndarray, sigma: float, torch_dtype=None, cp_dtype=None
    ) -> xp.ndarray:
        r"""Apply the distributed denoiser.

        Parameters
        ----------
        input_image: xp.ndarray
            Input image tile.
        sigma: float
            Denoiser parameter (noise standard deviation).
        torch_dtype : torch.dtype or None, optional
            Numerical precision to be used for computations with `torch`. Default is `None`.
        cp_dtype : xp.dtype or None, optional
            Numerical precision to be used for computations with `xp` (`numpy`
            or `cupy`). Default is `None`.

        Returns
        -------
        xp.ndarray
            Denoised image tile.
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
        r"""Apply the DDFB denoiser without communication on the first layer.

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
            Denoised image facet.
        """
        # TODO: see if accommodating xp.float* or torch.float*
        assert isinstance(sigma, float)
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
    def get_recv_size(self) -> xp.ndarray:
        return self.mpi_conv.direct_communicator.cartslicer.recv_size

    @property
    def get_send_size(self) -> xp.ndarray:
        return self.mpi_conv.direct_communicator.cartslicer.send_size

    @property
    def global_to_tile_slice(self):
        return self.mpi_conv.direct_communicator.cartslicer.slice_global_buffer_to_tile
