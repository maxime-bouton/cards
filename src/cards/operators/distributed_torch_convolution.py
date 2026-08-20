"""Distributed implementation of a pytorch linear convolution operator.

The class is leveraged in the distributed implementation of denoisers encoded
by convolutional neural networks (see :mod:`cards.denoisers`)
"""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from typing import Any

import numpy as np
import torch
from mpi4py import MPI

import cards.backend as xp
import cards.communicators.sync_cartesian_communicator as comms
from cards.operators.linear_operator import LinearOperator
from cards.utils.utils import torch2xp, xp2torch

# FIXME: find cleaner approach to accommodate conv/adj_conv in forward and adjoint methods while compatiable byth LinearOperator class


def slice_valid_coefficients(
    ranknd: np.ndarray,
    grid_size: np.ndarray,
    padding_size: np.ndarray,
):
    ndims = ranknd.size

    if not (grid_size.size == ndims and padding_size.size == ndims):
        raise AssertionError(
            r"`ranknd`, `grid_size` and `padding_size` must have the same shape"
        )

    L = padding_size * (ranknd > 0)
    R = -padding_size * (ranknd < grid_size - 1)

    return tuple(np.s_[i or None : r or None] for i, r in zip(L, R))


def slice_input2buffer(
    ranknd: np.ndarray,
    grid_size: np.ndarray,
    overlap_size: np.ndarray,
    backward: bool = False,
):
    ndims = ranknd.size

    if not (grid_size.size == ndims and overlap_size.size == ndims):
        raise AssertionError(
            r"`ranknd`, `grid_size` and `overlap_size` must have the same shape"
        )

    if backward:
        L = overlap_size * (ranknd > 0)
        return tuple([np.s_[i or None :] for i in L])
    else:
        R = -overlap_size * (ranknd < grid_size - 1)
        return tuple([np.s_[: r or None] for r in R])


class DistributedTorchConvolution(LinearOperator):
    r"""Synchronous distributed implementation of a `torch.Conv2d` operator in
    the `same` padding mode.

    Parameters
    ----------
    image_shape : tuple[int, ...]
        Full image size.
    kernel_shape : tuple[int, ...]
        Convolution kernel size.
    padding : tuple[int, int]
        Padding size to be applied to the image.
    comm : mpi4py.MPI.Comm
        Underlying MPI communicator.
    grid_shape : Sequence[int]
        Number of MPI workers along each of axis of the communicator grid.
    Cout : int, optional
        Number of channels in the output data. By default 0.
    tile_range : np.ndarray | None, optional
        Range of global indices defining the local image tile handled by the current worker, by default None. The global array is divided evenly across the different workers when ``tile_range`` is ``None``.
    backward : bool, optional
        Direction of the overlap between facets along all the axis for the direct operator (True for backward overlap, False for forward overlap). By default False.
    enable_internal_buffer : bool, optional
        Flag to enable interal storage of temporary buffers required for communication, by default True.

    Raises
    ------
    ValueError
        ``image_size`` and ``data_size`` must have the same number of
        elements.
    ValueError
        ``kernel`` should have ``ndims = len(image_size)`` dimensions.
    TypeError
        Only real-valued kernel supported.

    Attributes
    ----------
    dtype : type, optional
        Type of the entries in communicated arrays (np.float32).
    image_size : np.ndarray[int]
        Numpy array created from ``self.image_shape``.
    data_size : np.ndarray[int]
        Numpy array created from ``self.data_shape``.
    kernel_size : np.ndarray[int]
        Numpy array created from ``self.kernel_shape``.
    grid_size : np.ndarray[int]
        Numpy array created from ``grid_shape``.
    padding : np.ndarray(int)
        Padding size to be applied to the image.
    overlap_size : np.ndarray[int]
        Size of the overlap between contiguous facets along each axis of the problem.
    slice_valid_coefficients : Slice
        Slice to extract valid coefficients from the local forward/adjoint convolution.
    direct_communicator : cards.communicator.sync_cartesian_communicator.SyncCartesianCommunicator
        Communicator object to operate the communications required
        by the distributed implementation of the direct convolution operator.
    adjoint_communicator : cards.communicator.sync_cartesian_communicator.SyncCartesianCommunicator
        Communicator object to operate the communications required
        by the distributed implementation of the adjoint convolution operator.
    """

    def __init__(
        self,
        image_shape: tuple[int, ...],
        kernel_shape: tuple[int, ...],
        padding: tuple[int, ...],
        comm: MPI.Comm,
        grid_shape: tuple[int, ...],
        Cout: int = 0,
        backward=False,
        tile_range: np.ndarray | None = None,
        enable_internal_buffer: bool = True,
    ):
        self.image_size = np.asarray(image_shape)
        self.kernel_size = np.array((1,) * (self.image_size.size - 2) + kernel_shape)
        self.padding = np.array((0,) * (self.image_size.size - 2) + padding)
        self.data_size = self.image_size + 2 * self.padding - self.kernel_size + 1
        if not self.image_size.size == self.data_size.size:
            raise ValueError(
                "image_size and data_size must have the same number of elements"
            )
        if self.data_size.size > 2 and Cout:
            self.data_size[-3] = Cout
        super().__init__(image_shape, (*self.data_size,))

        self.dtype = np.float32
        self.grid_size = np.asarray(grid_shape)
        self.overlap_size = self.kernel_size - 1

        # * communicator for the distributed direct operator
        self.direct_communicator = comms.SyncCartesianCommunicator(
            comm,
            self.grid_size,
            self.image_size,
            self.overlap_size,
            self.overlap_size,
            backward=backward,
            dtype=self.dtype,
            tile_range=tile_range,
        )

        # slice to extract valid coefficients from the local forward
        # convolution output
        self.slice_valid_coefficients = slice_valid_coefficients(
            self.direct_communicator.ranknd,
            self.grid_size,
            self.padding,
        )

        # * communicator for the distributed adjoint operator
        # ! defining adjoint based on indices of the global convolution (output) to be handled on the current process

        # base case, valid when (grid_size == 1)
        local_data_size = (
            self.direct_communicator.cartslicer.tile_size
            + 2 * self.padding
            - self.overlap_size
        )

        # when distributed (grid_size > 1)
        local_data_size -= self.padding * (self.direct_communicator.ranknd > 0)
        local_data_size -= self.padding * (
            self.direct_communicator.ranknd < self.grid_size - 1
        )

        if backward:
            local_data_size += self.overlap_size * (self.direct_communicator.ranknd > 0)
            offset_id = (self.direct_communicator.ranknd > 0) * (
                self.padding - self.overlap_size
            )
        else:
            local_data_size += self.overlap_size * (
                self.direct_communicator.ranknd < self.grid_size - 1
            )
            offset_id = (self.direct_communicator.ranknd > 0) * self.padding

        if local_data_size.size > 2 and Cout:
            local_data_size[-3] = Cout

        tile_data = np.zeros((self.ndims, 2), dtype="i")
        tile_data[:, 0] = (
            self.direct_communicator.cartslicer.tile_range[:, 0] + offset_id
        )
        # id of last point in the data tile
        tile_data[:, 1] = tile_data[:, 0] + local_data_size - 1

        self.adjoint_communicator = comms.SyncCartesianCommunicator(
            comm,
            self.grid_size,
            self.data_size,
            self.overlap_size,
            self.overlap_size,
            backward=not backward,
            dtype=self.dtype,
            tile_range=tile_data,
        )

        if enable_internal_buffer:
            self.forward_buffer = xp.zeros(
                self.direct_communicator.cartslicer.facet_size, dtype=xp.float32
            )
        self.adjoint_buffer = xp.zeros(
            self.adjoint_communicator.cartslicer.facet_size, dtype=xp.float32
        )

        self.forward_input_slice = slice_input2buffer(
            self.direct_communicator.ranknd,
            self.grid_size,
            self.overlap_size,
            backward=backward,
        )
        self.adjoint_input_slice = slice_input2buffer(
            self.direct_communicator.ranknd,
            self.grid_size,
            self.overlap_size,
            backward=not backward,
        )

    def forward(self, image: xp.ndarray, op: torch.nn.Conv2d | None = None):
        r"""Implementation of the direct operator to update the input array
        ``image`` (from image to data space).

        Parameters
        ----------
        image : ndarray[float]
            Input buffer array (image space), of size ``self.direct_communicator.cartslicer.tile_size``.
        op : torch.nn.Conv2d (Callable[[torch.Tensor], torch.Tensor])
            Torch convolution operator.

        Returns
        -------
        ndarray[float]
            Result of the direct operator using the information from the local
            image facet.

        Note
        ----
        The input buffer ``image`` is copied inside forward_buffer, on GPU. This intern buffer will be used for the communications and the computations.
        """

        self.forward_buffer[self.forward_input_slice] = image
        self.direct_communicator.update_borders(self.forward_buffer)
        with torch.no_grad():
            return torch2xp(op(xp2torch(self.forward_buffer)))[
                self.slice_valid_coefficients
            ]

    def forward_no_comm(self, input_array: xp.ndarray, conv: torch.nn.Conv2d) -> Any:
        """forward_no_comm Apply the convolution whitout using the internal buffer. Should be used with a shared buffer.

        Parameters
        ----------
        input_array : xp.ndarray
            Input array, will not be updated to receive information from other thread. Facet size.
        conv : torch.nn.Conv2d

        Returns
        -------
        Any
            Result of the convolution
        """
        with torch.no_grad():
            return torch2xp(conv(xp2torch(input_array)))[self.slice_valid_coefficients]

    def adjoint(
        self,
        data: xp.ndarray,
        adjoint_op: torch.nn.Conv2d | torch.nn.ConvTranspose2d | None = None,
    ):
        r"""Implementation of the adjoint operator to update the input array
        ``data`` (from data to image space).

        Parameters
        ----------
        data : ndarray[float]
            Input buffer array (data space), of size ``self.adjoint_communicator.cartslicer.tile_size``.
        adjoint_op : torch.nn.ConvTranspose2d (Callable[[torch.Tensor], torch.Tensor])
            Torch adjoint convolution operator.

        Returns
        -------
        x : ndarray[float]
            Result of the adjoint operator using the information from the local
            data facet.

        Note
        ----
        The input is copied inside adjoint_buffer, on GPU. This intern buffer will be used for the communications and the computations.
        """

        self.adjoint_buffer[self.adjoint_input_slice] = data
        self.adjoint_communicator.update_borders(self.adjoint_buffer)
        with torch.no_grad():
            return torch2xp(adjoint_op(xp2torch(self.adjoint_buffer)))[
                self.slice_valid_coefficients
            ]

    def get_recv_size(self) -> np.ndarray:
        return self.direct_communicator.cartslicer.recv_size

    def get_send_size(self) -> np.ndarray:
        return self.direct_communicator.cartslicer.send_size
