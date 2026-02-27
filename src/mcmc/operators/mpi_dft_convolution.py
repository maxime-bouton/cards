"""Distributed implementation of the convolution product as a linear operator, computed via the DFT.
The computations can be done either on CPU or GPU depending on the settings.
"""

import numpy as np
from mpi4py import MPI

import mcmc.communicator.sync_cartesian_communicator as comms
from mcmc.backend import xp
from mcmc.operators.linear_operator import LinearOperator
from mcmc.operators.utils_convolution import fft_conv


def slice_valid_direct_convolution(ranknd, grid_size, overlap_size):
    r"""Helper function to extract the valid coefficients from the local
    convolution output.

    Returns a slice to select the valid local convolution coefficients for the
    direct convolution operator. Ensures a necessary padding to implement the adjoint (zero-padding) operator.

    Parameters
    ----------
    ranknd : numpy.ndarray[int]
        Rank of the process in a Cartesian nD grid of MPI processes.
    grid_size : numpy.ndarray[int]
        Size of the MPI process grid.
    overlap_size : numpy.ndarray[int]
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

    valid_coefficients = tuple([np.s_[L[d] : R[d]] for d in range(ndims)])

    return valid_coefficients


def slice_input2buffer_forward(ranknd, grid_size, overlap_size):
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

    valid_coefficients = tuple([np.s_[0 : R[d]] for d in range(ndims)])

    return valid_coefficients


def slice_input2buffer_adjoint(ranknd, grid_size, overlap_size):
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

    valid_coefficients = tuple([np.s_[L[d] : None] for d in range(ndims)])

    return valid_coefficients


class MpiDftConvolution(LinearOperator):
    r"""Synchronous distributed implementation of a linear convolution operator.

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
    kernel : cupy.ndarray[float]
        Input convolution kernel.
    backward : bool, optional
        Direction of the overlap between facets along all the axis (True
        for backward overlap, False for forward overlap). By default False.
    overlap_size : numpy.ndarray[int]
        Size of the overlap between contiguous facets along each of the ``d``
        axes of the problem.
    direct_conv_size : numpy.ndarray[int]
        Size of the local forward convolution to be performed on the process.
    adjoint_conv_size : numpy.ndarray[int]
        Size of the local adjoint convolution to be performed on the process.
    direct_fft_kernel : cupy.ndarray[float]
        Fourier transform of the convolution kernel for the local forward convolution.
    adjoint_fft_kernel : cupy.ndarray[float]
        Fourier transform of the convolution kernel for the local forward convolution.
    slice_valid_direct_convolution : Slice
        Slice to extract valid coefficients from the local forward convolution.
    slice_valid_adjoint_convolution : Slice
        Slice to extract valid coefficients from the local forward convolution.
    direct_communicator : dsgs.experimental.communicators.SyncCartesianCommunicator
        Communicator object to operate the communications required
        by the distributed implementation of the direct convolution operator.
    adjoint_communicator : dsgs.experimental.communicators.SyncCartesianCommunicator
        Communicator object to operate the communications required
        by the distributed implementation of the adjoint convolution operator.
    """

    def __init__(
        self,
        image_size: np.ndarray,
        kernel: xp.ndarray,
        comm: MPI.Comm,
        grid_size: xp.ndarray,
        backward=False,
        dtype: xp.dtype = xp.float64,
        tile_range: np.ndarray | None = None,
    ):
        r"""Synchronous distributed implementation of a (linear) convolution
        model.

        Parameters
        ----------
        image_size : numpy.ndarray[int], of size ``d``
            Full image size.
        kernel : numpy.ndarray[float] (real)
            Input convolution kernel. Only real-valued kernel are supported for
            now.
        comm : mpi4py.MPI.Comm
            Underlying MPI communicator.
        grid_size : numpy.ndarray[int]
            Number of workers along each of the ``d`` dimensions of the
            communicator grid.
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
        data_size = image_size + np.array([*kernel.shape], dtype="i") - 1
        if not image_size.size == data_size.size:
            raise ValueError(
                "image_size and data_size must have the same number of elements"
            )

        super(MpiDftConvolution, self).__init__(image_size, data_size)
        self.grid_size = grid_size
        self.comm = comm
        self.rank = self.comm.Get_rank()

        # * Cartesian communicator and nd rank
        self.cartcomm = self.comm.Create_cart(
            dims=self.grid_size,
            periods=self.ndims * [False],
            reorder=False,
        )
        self.ranknd = np.array(self.cartcomm.Get_coords(self.rank), dtype="i")

        # * useful dimensions
        if not len(kernel.shape) == self.ndims:
            raise ValueError("kernel should have ndims = len(image_size) dimensions")
        if kernel.dtype.kind == "c":
            raise TypeError("only real-valued kernel supported")
        self.overlap_size = np.array(kernel.shape, dtype="i") - 1

        # * communicator for the distributed direct operator
        self.direct_communicator = comms.SyncCartesianCommunicator(
            self.comm,
            self.grid_size,
            self.image_size,
            self.overlap_size,
            self.overlap_size,
            dtype=dtype,
            backward=backward,
            tile_range=tile_range,
        )
        # kernel and slice to extract valid coefficients from the local forward
        # convolution output
        self.direct_conv_size = tuple(
            self.direct_communicator.cartslicer.facet_size + self.overlap_size
        )
        self.direct_fft_kernel = xp.fft.rfftn(kernel, self.direct_conv_size)

        self.slice_valid_direct_convolution = slice_valid_direct_convolution(
            self.ranknd, self.grid_size, self.overlap_size
        )

        # * communicator for the distributed adjoint operator
        # ! defining adjoint based on indices of the global convolution (output) to be handled on the current process
        if backward:
            local_data_size = (
                self.direct_communicator.cartslicer.tile_size
                + (self.ranknd == grid_size - 1) * self.overlap_size
            )
            offset_id = 0
        else:
            local_data_size = (
                self.direct_communicator.cartslicer.tile_size
                + (self.ranknd == 0) * self.overlap_size
            )
            offset_id = (self.ranknd > 0) * self.overlap_size

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
            dtype=dtype,
            backward=not backward,
            tile_range=tile_data,
        )

        # kernel and slice to extract valid coefficients from the local adjoint
        # convolution output
        self.adjoint_conv_size = tuple(
            self.adjoint_communicator.cartslicer.facet_size + self.overlap_size
        )
        self.adjoint_fft_kernel = xp.conj(xp.fft.rfftn(kernel, self.adjoint_conv_size))
        self.slice_valid_adjoint_convolution = tuple(
            [
                np.s_[: self.direct_communicator.cartslicer.tile_size[d]]
                for d in range(self.ndims)
            ]
        )

        self.forward_buffer = xp.zeros(
            self.direct_communicator.cartslicer.facet_size,
            dtype=dtype,
        )
        self.adjoint_buffer = xp.zeros(
            self.adjoint_communicator.cartslicer.facet_size,
            dtype=dtype,
        )

        self.forward_input_slice = slice_input2buffer_forward(
            self.ranknd, self.grid_size, self.overlap_size
        )
        self.adjoint_input_slice = slice_input2buffer_adjoint(
            self.ranknd, self.grid_size, self.overlap_size
        )

    def forward(self, input_image):
        r"""Implementation of the direct operator to update the input array
        ``input_image`` (from image to data space).

        Parameters
        ----------
        input_image : ndarray[float]
            Input buffer array (image space), of size ``self.direct_communicator.cartslicer.tile_size``.

        Returns
        -------
        y : ndarray[float]
            Result of the direct operator using the information from the local
            image facet.

        Note
        ----
        The input buffer ``input_image`` is copied inside forward_buffer, on GPU. This intern buffer will be used for the communications and the computations.
        """

        self.forward_buffer[self.forward_input_slice] = input_image
        self.direct_communicator.update_borders(self.forward_buffer)
        y = fft_conv(
            self.forward_buffer,
            self.direct_fft_kernel,
            self.direct_conv_size,
        )[self.slice_valid_direct_convolution]
        return y

    def adjoint(self, input_data):
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
        x = fft_conv(
            self.adjoint_buffer,
            self.adjoint_fft_kernel,
            self.adjoint_conv_size,
        )[self.slice_valid_adjoint_convolution]
        return x
