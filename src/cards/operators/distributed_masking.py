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
        Mask tensor for the current worker, with 0 corresponding to masked entries, 1 to observed entries.

    Attributes
    ----------
    mask_tile : xp.ndarray
        Mask tensor, with 0 corresponding to masked entries, 1 to observed entries.
    dtype : type, optional
        Type of the entries in communicated arrays, by default xp.float64.
    image_size : np.ndarray[int]
        Numpy array created from ``self.image_shape``.
    data_size : np.ndarray[int]
        Numpy array created from ``self.data_shape``.
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

        if not len(mask_tile.shape) == self.ndims:
            raise ValueError("mask should have ndims = len(image_size) dimensions")

        if not (
            self.cartslicer.tile_range is not None
            and np.allclose(
                np.asarray(mask_tile)[None, ...],
                np.diff(self.cartslicer.tile_range, axis=-1),
            )
        ):
            raise ValueError(
                "mask_tile shape is not consistent with cartslicer.tile_range"
            )
