import numpy as np
import torch
from mpi4py import MPI

import mcmc.communicator.sync_cartesian_communicator as comms
from mcmc.backend import xp
from mcmc.operators.linear_operator import LinearOperator
from mcmc.utils.utils import torch2xp, xp2torch


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

    return tuple(np.s_[l or None : r or None] for l, r in zip(L, R))


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
        return tuple([np.s_[l or None :] for l in L])
    else:
        R = -overlap_size * (ranknd < grid_size - 1)
        return tuple([np.s_[: r or None] for r in R])


class MpiTorchConvolution(LinearOperator):
    r"""Synchronous distributed implementation of a `torch.Conv2d` operator in the
    `same` padding mode.

    Attributes
    ----------
    comm : mpi4py.MPI.Comm
        Underlying MPI communicator.
    cartcomm : mpi4py.MPI.Cartcomm
        Cartesian MPI communicator underlying the communications.
    rank : int
        Rank of the current MPI-process.
    ranknd : numpy.ndarray[int]
        Multi-linear rank of the current MPI-process in the Cartesian grid of
        workers (nD setting).
    grid_size : list of int, of size ``d``
        Number of workers along each of the ``d`` dimensions of the
        communicator grid.
    overlap_size : numpy.ndarray[int]
        Size of the overlap between contiguous facets along each of the ``d``
        axes of the problem.
    slice_valid_coefficients : Slice
        Slice to extract valid coefficients from the local forward/adjoint convolution.
    direct_communicator : mcmc.communicator.sync_cartesian_communicator.SyncCartesianCommunicator
        Communicator object to operate the communications required
        by the distributed implementation of the direct convolution operator.
    adjoint_communicator : mcmc.communicator.sync_cartesian_communicator.SyncCartesianCommunicator
        Communicator object to operate the communications required
        by the distributed implementation of the adjoint convolution operator.
    """

    def __init__(
        self,
        image_size: np.ndarray,
        kernel_size: tuple[int, ...],
        padding: tuple[int, ...],
        comm: MPI.Comm,
        grid_size: np.ndarray,
        Cout: int = 0,
        backward=False,
        tile_range: np.ndarray | None = None,
    ):
        r"""Synchronous distributed implementation of a `torch.Conv2d` operator in the
        `same` padding mode.

        Parameters
        ----------
        image_size : numpy.ndarray[int], of size ``d``
            Full image size.
        kernel_size : tuple[int, int]
            Convolution kernel size.
        padding : tuple[int, int]
            Padding size to be applied to the image.
        comm : mpi4py.MPI.Comm
            Underlying MPI communicator.
        grid_size : numpy.ndarray[int]
            Number of workers along each of the ``d`` dimensions of the
            communicator grid.
        Cout : int, optional
            Number of channels in the output data. By default 0.
        backward : bool, optional
            Direction of the overlap between facets along all the axis for the direct operator (True for backward overlap, False for forward overlap). By default False.

        Raises
        ------
        ValueError
            ``image_size`` and ``data_size`` must have the same number of
            elements.
        ValueError
            ``kernel`` should have ``ndims = len(image_size)`` dimensions.
        TypeError
            Only real-valued kernel supported.
        """
        self.kernel_size = np.array((1,) * (image_size.size - 2) + kernel_size)
        self.padding = np.array((0,) * (image_size.size - 2) + padding)
        data_size = image_size + 2 * self.padding - self.kernel_size + 1
        if not image_size.size == data_size.size:
            raise ValueError(
                "image_size and data_size must have the same number of elements"
            )
        if data_size.size > 2 and Cout:
            data_size[-3] = Cout
        super().__init__(image_size, data_size)
        self.grid_size = grid_size
        self.comm = comm
        self.rank = self.comm.Get_rank()

        # * Cartesian communicator and nd rank
        self.cartcomm = self.comm.Create_cart(
            dims=grid_size,
            periods=self.ndims * [False],
            reorder=False,
        )
        self.ranknd = np.array(self.cartcomm.Get_coords(self.rank), dtype="i")

        self.overlap_size = self.kernel_size - 1

        # * communicator for the distributed direct operator
        self.direct_communicator = comms.SyncCartesianCommunicator(
            self.comm,
            self.grid_size,
            self.image_size,
            self.overlap_size,
            self.overlap_size,
            backward=backward,
            dtype=np.float32,
            tile_range=tile_range,
        )
        # kernel and slice to extract valid coefficients from the local forward
        # convolution output

        self.slice_valid_coefficients = slice_valid_coefficients(
            self.ranknd,
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
        local_data_size -= self.padding * (self.ranknd > 0)
        local_data_size -= self.padding * (self.ranknd < grid_size - 1)

        if backward:
            local_data_size += self.overlap_size * (self.ranknd > 0)
            offset_id = (self.ranknd > 0) * (self.padding - self.overlap_size)
        else:
            local_data_size += self.overlap_size * (self.ranknd < grid_size - 1)
            offset_id = (self.ranknd > 0) * self.padding

        if local_data_size.size > 2 and Cout:
            local_data_size[-3] = Cout

        tile_data = np.zeros((self.ndims, 2), dtype="i")
        tile_data[:, 0] = (
            self.direct_communicator.cartslicer.tile_range[:, 0] + offset_id
        )
        # id of last point in the data tile
        tile_data[:, 1] = tile_data[:, 0] + local_data_size - 1

        self.adjoint_communicator = comms.SyncCartesianCommunicator(
            self.comm,
            self.grid_size,
            self.data_size,
            self.overlap_size,
            self.overlap_size,
            backward=not backward,
            dtype=np.float32,
            tile_range=tile_data,
        )

        self.forward_buffer = xp.zeros(
            self.direct_communicator.cartslicer.facet_size, dtype=xp.float32
        )
        self.adjoint_buffer = xp.zeros(
            self.adjoint_communicator.cartslicer.facet_size, dtype=xp.float32
        )

        self.forward_input_slice = slice_input2buffer(
            self.ranknd, self.grid_size, self.overlap_size, backward=backward
        )
        self.adjoint_input_slice = slice_input2buffer(
            self.ranknd, self.grid_size, self.overlap_size, backward=not backward
        )

    def forward(self, input_image: xp.ndarray, conv: torch.nn.Conv2d):
        r"""Implementation of the direct operator to update the input array
        ``input_image`` (from image to data space).

        Parameters
        ----------
        input_image : ndarray[float]
            Input buffer array (image space), of size ``self.direct_communicator.cartslicer.tile_size``.

        Returns
        -------
        ndarray[float]
            Result of the direct operator using the information from the local
            image facet.

        Note
        ----
        The input buffer ``input_image`` is copied inside forward_buffer, on GPU. This intern buffer will be used for the communications and the computations.
        """

        self.forward_buffer[self.forward_input_slice] = input_image
        self.direct_communicator.update_borders(self.forward_buffer)
        with torch.no_grad():
            return torch2xp(conv(xp2torch(self.forward_buffer)))[
                self.slice_valid_coefficients
            ]

    def adjoint(self, input_data: xp.ndarray, conv_adj: torch.nn.ConvTranspose2d):
        r"""Implementation of the adjoint operator to update the input array
        ``input_data`` (from data to image space).

        Parameters
        ----------
        input_data : ndarray[float]
            Input buffer array (data space), of size ``self.adjoint_communicator.cartslicer.tile_size``.

        Returns
        -------
        x : ndarray[float]
            Result of the adjoint operator using the information from the local
            data facet.

        Note
        ----
        The input is copied inside adjoint_buffer, on GPU. This intern buffer will be used for the communications and the computations.
        """

        self.adjoint_buffer[self.adjoint_input_slice] = input_data
        self.adjoint_communicator.update_borders(self.adjoint_buffer)
        with torch.no_grad():
            return torch2xp(conv_adj(xp2torch(self.adjoint_buffer)))[
                self.slice_valid_coefficients
            ]
