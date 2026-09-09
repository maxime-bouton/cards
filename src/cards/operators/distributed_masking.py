"""Implementation of the masking operator involved in inpainting problems."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

import numpy as np

import cards.backend as xp
from cards.operators.linear_operator import LinearOperator
from cards.slicers.cartesian_comm_slicer import CartesianCommSlicer

# from mpi4py import MPI
# from cards.communicators.mpi_utils import get_ranknd

# FIXME: factorize implementation of DistributedLinearOperator (only a matter of adding an execution context on the abstract level and Slicer to a LinearOperator)
# -> better factorization between operators
# take inspiration from BaseDistributedModel to add a few methods/cached properties


class DistributedMasking(LinearOperator):
    r"""Implementation of a distributed masking operator as involved in inpainting problems.

    Parameters
    ----------
    mask_tile : xp.ndarray
        Local mask tensor tile for the current worker, with 0 corresponding to masked entries, 1 to observed entries.

    Attributes
    ----------
    mask : xp.ndarray
        Local mask tensor tile for the current worker, with 0 corresponding to masked entries, 1 to observed entries.
    cartslicer : CartesianCommSlicer
        Slicer describing the Cartesian tensor tessellation adopted for data distribution.
    grid_size : np.ndarray[int]
        Numpy array created from ``grid_shape``.

    Note
    ----
    Masking is implemented as an Hadamard product, and not as a cropping operator (i.e., retaining only non-masked entries from an input tensor).
    """

    def __init__(
        self,
        grid_shape: tuple[int, ...],
        image_shape: tuple[int, ...],
        mask_tile: xp.ndarray,
        cartslicer: CartesianCommSlicer,
    ):
        super().__init__(image_shape, image_shape)
        self.grid_size = np.asarray(grid_shape)
        self.cartslicer = cartslicer
        self.mask = mask_tile

        if not len(mask_tile.shape) == self.ndims:
            raise ValueError("mask should have ndims = len(image_size) dimensions")

        if not (
            self.cartslicer.tile_range is not None
            and np.allclose(
                np.squeeze(np.asarray(mask_tile.shape)),
                np.squeeze(np.diff(self.cartslicer.tile_range, axis=-1)) + 1,
            ),
        ):
            raise ValueError(
                "mask_tile shape is not consistent with cartslicer.tile_range"
            )

    def forward(self, image: xp.ndarray, op=None) -> xp.ndarray:
        return self.mask * image

    def adjoint(self, data: xp.ndarray, adjoint_op=None) -> xp.ndarray:
        return self.mask * data
