"""Distributed implementation of an FFT-based convolution operator."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from collections.abc import Sequence

import numpy as np
from mpi4py import MPI

import cards.backend as xp
import cards.communicators.sync_cartesian_communicator as comms
from cards.operators.dft_convolution import fft_conv
from cards.operators.linear_operator import LinearOperator

# TODO: check slices, does not currently rely on the slicer included in the communicator : use a Slicer instead?
# FIXME: reduce redundancy between slice generating functions below


def slice_valid_direct_convolution(
    ranknd: np.ndarray, grid_size: np.ndarray, overlap_size: np.ndarray
) -> tuple[slice]:
    r"""Helper function to extract the valid coefficients from the local
    convolution output.

    Returns a slice to select the valid local convolution coefficients for the
    direct convolution operator. Ensures a necessary padding to implement the adjoint (zero-padding) operator.

    Parameters
    ----------
    ranknd : np.ndarray[int]
        Rank of the process in a Cartesian nD grid of MPI processes.
    grid_size : np.ndarray[int]
        Number of MPI workers along each of axis of the communicator grid.
    overlap_size : np.ndarray[int]
        Overlap between contiguous facets along each dimension.

    Returns
    -------
    valid_coefficients : tuple[slice]
        Slice to extract valid coefficients from the local convolutions.

    Raises
    ------
    AssertionError
        `ranknd`, `grid_size` and `overlap_size` must all have the save shape.
    """

    ndims = ranknd.size

    if not (grid_size.size == ndims and overlap_size.size == ndims):
        raise AssertionError(
            r"`ranknd`, `grid_size` and `overlap_size` must have the save \
                shape"
        )

    L = ndims * [None]
    R = ndims * [None]

    for d in range(ndims):
        if grid_size[d] > 1 and overlap_size[d] > 0:
            if ranknd[d] > 0 and ranknd[d] < grid_size[d] - 1:
                L[d] = overlap_size[d]
                R[d] = -overlap_size[d]
            elif ranknd[d] == grid_size[d] - 1:
                L[d] = overlap_size[d]
                R[d] = None
            else:
                L[d] = 0
                R[d] = -overlap_size[d]
        else:
            L[d] = 0
            R[d] = None

    valid_coefficients = tuple([xp.s_[L[d] : R[d]] for d in range(ndims)])

    return valid_coefficients


def slice_input2buffer_forward(
    ranknd: np.ndarray, grid_size: np.ndarray, overlap_size: np.ndarray
) -> tuple[slice]:
    r"""Generate slices to place the local image-tile into the local image-facet buffer for the forward operator.

    Parameters
    ----------
    ranknd : np.ndarray
        Multi-linear rank of the current MPI-process in the Cartesian grid of
        workers.
    grid_size : np.ndarray
        Number of MPI workers along each of axis of the communicator grid.
    overlap_size : np.ndarray
        Size of the overlap between contiguous image facets along each axis.

    Returns
    -------
    tuple[slice]
        Output slices.

    Raises
    ------
    AssertionError
        `ranknd`, `grid_size` and `overlap_size` must have the save shape
    """
    ndims = ranknd.size

    if not (grid_size.size == ndims and overlap_size.size == ndims):
        raise AssertionError(
            r"`ranknd`, `grid_size` and `overlap_size` must have the save \
                shape"
        )

    R = ndims * [None]

    for d in range(ndims):
        if grid_size[d] > 1 and overlap_size[d] > 0:
            if ranknd[d] > 0 and ranknd[d] < grid_size[d] - 1:
                R[d] = -overlap_size[d]
            elif ranknd[d] == grid_size[d] - 1:
                R[d] = None
            else:
                R[d] = -overlap_size[d]
        else:
            R[d] = None

    valid_coefficients = tuple([xp.s_[0 : R[d]] for d in range(ndims)])

    return valid_coefficients


def slice_input2buffer_adjoint(
    ranknd: np.ndarray, grid_size: np.ndarray, overlap_size: np.ndarray
) -> tuple[slice]:
    r"""Generate slices to place the local data-tile into the local data-facet buffer for the forward operator.

    Parameters
    ----------
    ranknd : np.ndarray
        Multi-linear rank of the current MPI-process in the Cartesian grid of
        workers.
    grid_size : np.ndarray
        Number of MPI workers along each of axis of the communicator grid.
    overlap_size : np.ndarray
        Size of the overlap between contiguous image facets along each axis.

    Returns
    -------
    tuple[slice]
        Output slices.

    Raises
    ------
    AssertionError
        `ranknd`, `grid_size` and `overlap_size` must have the save shape
    """
    ndims = ranknd.size

    if not (grid_size.size == ndims and overlap_size.size == ndims):
        raise AssertionError(
            r"`ranknd`, `grid_size` and `overlap_size` must have the save \
                shape"
        )

    L = ndims * [None]

    for d in range(ndims):
        if grid_size[d] > 1 and overlap_size[d] > 0:
            if ranknd[d] > 0 and ranknd[d] > grid_size[d] - 1:
                L[d] = overlap_size[d]
            elif ranknd[d] == 0:
                L[d] = None
            else:
                L[d] = overlap_size[d]
        else:
            L[d] = None

    valid_coefficients = tuple([xp.s_[L[d] : None] for d in range(ndims)])

    return valid_coefficients


class DistributedDftConvolution(LinearOperator):
    r"""Synchronous distributed implementation of a linear convolution operator.

    Parameters
    ----------
    grid_shape : Sequence[int]
        Number of MPI workers along each of axis of the communicator grid.
    comm: mpi4py.MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD.
    kernel : xp.ndarray
        Input convolution kernel. Only real-valued kernels are supported for now.
    enable_internal_buffer : bool, optional
        Flag to enable interal storage of temporary buffers required for communication, by default True.
    dtype : type, optional
        Type of the entries in communicated arrays, by default xp.float64.
    tile_range : np.ndarray | None, optional
        Range of global indices defining the local image tile handled by the current worker, by default None. The global array is divided evenly across the different workers when ``tile_range`` is ``None``.
    backward : bool, optional
        Direction of the overlap between image facets handled by consecutive workers in the MPI grid for the forward operator, by default False.

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
        Type of the entries in communicated arrays, by default xp.float64.
    image_size : np.ndarray[int]
        Numpy array created from ``self.image_shape``.
    data_size : np.ndarray[int]
        Numpy array created from ``self.data_shape``.
    grid_size : np.ndarray[int]
        Numpy array created from ``grid_shape``.
    overlap_size : np.ndarray[int]
        Size of the overlap between contiguous facets along each of the ``d``
        axes of the problem.
    direct_communicator : cards.communicators.sync_cartesian_communicator.SyncCartesianCommunicator
        Communicator object to operate the communications required
        by the distributed implementation of the direct convolution operator.
    adjoint_communicator : cards.communicators.sync_cartesian_communicator.SyncCartesianCommunicator
        Communicator object to operate the communications required
        by the distributed implementation of the adjoint convolution operator.
    fft_kernel : xp.ndarray
        Fourier transform of the convolution kernel for the local forward convolution.
    adjoint_fft_kernel : xp.ndarray
        Fourier transform of the convolution kernel for the local forward convolution.
    direct_conv_size : np.ndarray[int]
        Size of the local forward convolution performed on the current worker.
    adjoint_conv_size : np.ndarray[int]
        Size of the local adjoint convolution performed on the current worker.
    slice_valid_direct_convolution : Slice
        Slice to extract valid coefficients from the local forward convolution.
    slice_valid_adjoint_convolution : Slice
        Slice to extract valid coefficients from the local forward convolution.
    forward_buffer : xp.ndarray
        Temporary buffer to receive communications involved in the forward operator.
    adjoint_buffer : xp.ndarray
        Temporary buffer to receive communications involved in the adjoint operator.
    forward_input_slice : Slice
        Slice to place the local image-tile into the local image-facet buffer for the forward operator.
    adjoint_input_slice : Slice
        Slice to place the local data-tile into the local data-facet buffer for the adjoint operator.
    """

    def __init__(
        self,
        image_shape: Sequence[int],
        grid_shape: xp.ndarray,
        comm: MPI.Comm,
        kernel: xp.ndarray,
        enable_internal_buffer: bool = True,
        dtype: type = xp.float64,
        tile_range: np.ndarray | None = None,
        backward: bool = False,
    ):
        self.image_size = np.asarray(image_shape)
        self.data_size = self.image_size + np.asarray(kernel.shape) - 1
        data_shape = (*self.data_size,)
        super().__init__(image_shape, data_shape)

        if not len(self.image_shape) == len(self.data_shape):
            raise ValueError(
                "image_size and data_size must have the same number of elements"
            )

        self.dtype = dtype
        self.grid_size = np.asarray(grid_shape)

        # * useful dimensions
        if not len(kernel.shape) == self.ndims:
            raise ValueError("kernel should have ndims = len(image_size) dimensions")
        # TODO: see if this is stil the case
        if kernel.dtype.kind == "c":
            raise TypeError("only real-valued kernel supported")
        self.overlap_size = np.asarray(kernel.shape) - 1

        # * communicator for the distributed direct operator
        self.direct_communicator = comms.SyncCartesianCommunicator(
            comm,
            self.grid_size,
            self.image_size,
            self.overlap_size,
            self.overlap_size,
            dtype=self.dtype,
            backward=backward,
            tile_range=tile_range,
        )
        self.grid_size = self.direct_communicator.grid_size

        # kernel and slice to extract valid coefficients from the local forward
        # convolution output
        self.direct_conv_size = tuple(
            self.direct_communicator.cartslicer.facet_size + self.overlap_size
        )
        self.fft_kernel = xp.fft.rfftn(
            kernel, self.direct_conv_size, axes=range(len(kernel.shape))
        )
        self.slice_valid_direct_convolution = slice_valid_direct_convolution(
            self.direct_communicator.ranknd, self.grid_size, self.overlap_size
        )

        # * communicator for the distributed adjoint operator
        # ! defining adjoint based on indices of the global convolution (output) to be handled on the current process
        if backward:
            local_data_size = (
                self.direct_communicator.cartslicer.tile_size
                + (self.direct_communicator.ranknd == self.grid_size - 1)
                * self.overlap_size
            )
            offset_id = 0
        else:
            local_data_size = (
                self.direct_communicator.cartslicer.tile_size
                + (self.direct_communicator.ranknd == 0) * self.overlap_size
            )
            offset_id = (self.direct_communicator.ranknd > 0) * self.overlap_size

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
            dtype=self.dtype,
            backward=not backward,
            tile_range=tile_data,
        )

        # kernel and slice to extract valid coefficients from the local adjoint
        # convolution output
        self.adjoint_conv_size = tuple(
            self.adjoint_communicator.cartslicer.facet_size + self.overlap_size
        )
        self.adjoint_fft_kernel = xp.conj(
            xp.fft.rfftn(kernel, self.adjoint_conv_size, axes=range(len(kernel.shape)))
        )
        self.slice_valid_adjoint_convolution = tuple(
            [
                np.s_[: self.direct_communicator.cartslicer.tile_size[d]]
                for d in range(self.ndims)
            ]
        )

        if enable_internal_buffer:
            self.forward_buffer = xp.zeros(
                self.direct_communicator.cartslicer.facet_size,
                dtype=dtype,
            )
        self.adjoint_buffer = xp.zeros(
            self.adjoint_communicator.cartslicer.facet_size,
            dtype=dtype,
        )

        self.forward_input_slice = slice_input2buffer_forward(
            self.direct_communicator.ranknd, self.grid_size, self.overlap_size
        )
        self.adjoint_input_slice = slice_input2buffer_adjoint(
            self.direct_communicator.ranknd, self.grid_size, self.overlap_size
        )

    def forward(self, image: xp.ndarray) -> xp.ndarray:
        # NOTE: in this function, ``image`` refers to the local image tile handled by the current process
        # NOTE: The input buffer ``input_image`` is copied inside forward_buffer, on GPU. This intern buffer will be used for the communications and the computations.
        self.forward_buffer[self.forward_input_slice] = image
        self.direct_communicator.update_borders(self.forward_buffer)
        y = fft_conv(
            self.forward_buffer,
            self.fft_kernel,
            self.direct_conv_size,
        )[self.slice_valid_direct_convolution]
        return y

    def adjoint(self, data: xp.ndarray) -> xp.ndarray:
        # NOTE: in this function, ``data`` refers to the local data tile handled by the current process
        # NOTE: The input is copied inside adjoint_buffer, on GPU. This intern buffer will be used for the communications and the computations.
        self.adjoint_buffer[self.adjoint_input_slice] = data
        self.adjoint_communicator.update_borders(self.adjoint_buffer)
        x = fft_conv(
            self.adjoint_buffer,
            self.adjoint_fft_kernel,
            self.adjoint_conv_size,
        )[self.slice_valid_adjoint_convolution]
        return x

    def forward_no_comm(self, image: xp.ndarray) -> xp.ndarray:
        # NOTE: in this function, ``image`` refers to the local image tile handled by the current process
        return fft_conv(image, self.fft_kernel, self.direct_conv_size)[
            self.slice_valid_direct_convolution
        ]

    # TODO: add this collection of features through inhteritance, instead of repeating it each time?
    def get_send_size(self) -> np.ndarray:
        return self.direct_communicator.cartslicer.send_size

    def get_recv_size(self) -> np.ndarray:
        return self.direct_communicator.cartslicer.recv_size
