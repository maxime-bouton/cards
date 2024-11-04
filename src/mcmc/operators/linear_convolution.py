"""Serial and distributed linear convolution operators."""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# reference: P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.

import numpy as np

import mcmc.utils.communications as ucomm
import mcmc.utils.communicators as comms
from mcmc.operators.convolutions import fft_conv
from mcmc.operators.distributed_convolutions import calculate_local_data_size
from mcmc.operators.linear_operator import LinearOperator

# ? = Keep kernel value out of the associated model object?
# TODO: - trim-down number of attributes in the classes?


# * Convolution model class (serial and distributed)
# ! keep kernel out of the class? (e.g., for blind deconvolution)
class SerialConvolution(LinearOperator):
    r"""Serial (FFT-based) convolution operator.

    Attributes
    ----------
    image_size : numpy.ndarray of int, of size ``d``
        Full image size.
    kernel : numpy.ndarray
        Input kernel. The array should have ``d`` axis, such that
        ``kernel.shape[i] < image_size[i]`` for ``i in range(d)``.
    data_size : empty numpy.ndarray of int, of size ``d``
        Full data size.
        - If ``data_size == image_size``: circular convolution;
        - If ``data_size == image_size + kernel_size - 1``: linear convolution.
    """

    # TODO: make sure the interface works for both linear and circular
    # TODO- convolutions
    def __init__(
        self,
        image_size,
        kernel,
        data_size,
    ):
        r"""SerialConvolution constructor.

        Parameters
        ----------
        image_size : numpy.ndarray of int, of size ``d``
            Full image size.
        kernel : numpy.ndarray of float
            Input kernel. The array should have ``d`` axis, such that
            ``kernel.shape[i] < image_size[i]`` for ``i in range(d)``.
        data_size : numpy.ndarray of int, of size ``d``
            Full data size.
            - If ``data_size == image_size``: circular convolution;
            - If ``data_size == image_size + kernel_size - 1``: linear convolution.
        fft_kernel : numpy.ndarray
            Fourier transform of the known convolution kernel.
        valid_coefficients : Slice
            Slice object to retrieve valid coefficients after applying the
            adjoint convolution operator.

        Raises
        ------
        ValueError
            ``image_size`` and ``data_size`` must have the same number of
            elements.
        ValueError
            ``kernel`` should have ``ndims = len(image_size)`` dimensions.
        TypeError
            Only real-valued kernel supported.

        Note
        ----
        Setting ``data_size`` to the same value as ``image_size`` results in a
        circular convolution.
        """
        if not image_size.size == data_size.size:
            raise ValueError(
                "image_size and data_size must have the same number of elements"
            )
        super(SerialConvolution, self).__init__(image_size, data_size)

        if not len(kernel.shape) == self.ndims:
            raise ValueError("kernel should have ndims = len(image_size) dimensions")

        if kernel.dtype.kind == "c":
            raise TypeError("only real-valued kernel supported")

        self.kernel = kernel
        self.fft_kernel = np.fft.rfftn(self.kernel, self.data_size)
        self.valid_coefficients = tuple(
            [np.s_[: self.image_size[d]] for d in range(self.ndims)]
        )

    def forward(self, input_image):
        r"""Implementation of the direct operator to update the input array
        ``input_image`` (from image to data space).

        Parameters
        ----------
        input_image : numpy.ndarray of float
            Input array (image space).

        Returns
        -------
        numpy.ndarray
            Convolution result (direct operator).
        """
        return fft_conv(input_image, self.fft_kernel, self.data_size)

    def adjoint(self, input_data):
        r"""Implementation of the adjoint operator to update the input array
        ``input_data`` (from data to image space).

        Parameters
        ----------
        input_data : numpy.ndarray of float
            Input array (data space).

        Returns
        -------
        numpy.ndarray
            Convolution result (adjoint operator).
        """
        return fft_conv(input_data, np.conj(self.fft_kernel), self.data_size)[
            self.valid_coefficients
        ]


# ! keep kernel out of the class?
class SyncLinearConvolution(LinearOperator):
    r"""Synchronous MPI-distributed implementation of a (linear) convolution
    operator.

    Attributes
    ----------
    image_size : numpy.ndarray of int, of size ``d``
        Full image size.
    data_size : numpy.ndarray of int, of size ``d``
        Full data size.
    kernel : numpy.ndarray of float
        Input convolution kernel.
    grid_size : list of int, of size ``d``
        Number of workers along each of the ``d`` dimensions of the
        communicator grid.
    itemsize : numpy.dtype.itemsize
        Size (in bytes) of the scalar type to be handled during the
        communications.
    circular_boundaries : bool
        Indicates whether periodic boundary conditions need to be
        considered for the communicator grid along each axis.
    backward : bool, optional
        Direction of the overlap between facets along all the axis (True
        for backward overlap, False for forward overlap). By default False.
    comm : mpi4py.MPI.Comm
        MPI communicator unferlying all the communications.
    cartcomm : mpi4py.MPI.Cartcomm
        Cartesian MPI communicator underlying the communications.
    rank : int
        Rank of the current MPI-process.
    ranknd : numpy.ndarray[int]
        Multi-linear rank of the current MPI-process in the Cartesian grid of
        workers (nD setting).
    overlap_size : numpy.ndarray[int]
        Size of the overlap between contiguous facets along each of the ``d``
        axes of the problem.
    tile_size : numpy.ndarray[int]
        Size of the non-overlapping tiles handled by each MPI process for the
        direct convolution operator.
    local_data_size : numpy.ndarray[int]
        Size of the data array handled by each MPI process.
    facet_size : numpy.ndarray of int, of size ``d``
        Number of elements along each of the ``d`` dimensions of the facet
        handled by the current process for application of the direct
        convolution operator.
    facet_size_adj : numpy.ndarray of int, of size ``d``
        Number of elements along each of the ``d`` dimensions of the facet
        handled by the current process for the adjoint convolution operator.
    local_conv_size : numpy.ndarray[int]
        Size of the local buffer storing the result of the convolution.
    offset : numpy.ndarray[int]
        Offset to retrieve the content of the tile used for
        the direct convolution operator from the facet (overlapping borders
        between contiguous workers).
    offset_adj : numpy.ndarray[int]
        Offset to retrieve the content of the tile used for the adjoint
        convolution operator from the adjoint facet (overlapping borders
        between contiguous workers).
    fft_kernel : numpy.ndarray[float]
        Fourier transform of the known convolution kernel.
    local_slice_valid_conv
        Slice object to extract valid coefficients after local convolutions.
    local_slice_conv
        Slice object to retrieve image tile from local fft-based convolution
        (adjoint conv. operator).
    local_slice_conv_adj
        Slice to set value of local convolution in the adjoint buffer.
    direct_communicator : dsgs.utils.communicators.SyncCartesianCommunicator
        Communicator object abstracting out the MPI-communications required
        by the distributed implementation of the direct convolution operator.
    adjoint_communicator : dsgs.utils.communicators.SyncCartesianCommunicator
        Communicator object abstracting out the MPI-communications required
        by the distributed implementation of the direct convolution operator.
    """

    def __init__(
        self,
        image_size,
        data_size,
        kernel,
        comm,
        grid_size,
        itemsize,
        circular_boundaries,
        backward=False,
    ):
        r"""Synchronous distributed implementation of a (linear) convolution
        model.

        Parameters
        ----------
        image_size : numpy.ndarray of int, of size ``d``
            Full image size.
        data_size : numpy.ndarray of int, of size ``d``
            Full data size.
        kernel : numpy.ndarray of float
            Input convolution kernel.
        comm : mpi4py.MPI.Comm
            Underlying MPI communicator.
        grid_size : list of int, of size ``d``
            Number of workers along each of the ``d`` dimensions of the
            communicator grid.
        itemsize : numpy.dtype.itemsize
            Size (in bytes) of the scalar type to be handled during the
            communications.
        circular_boundaries : bool
            Indicates whether periodic boundary conditions need to be
            considered for the communicator grid along each axis.
        backward : bool, optional
            Direction of the overlap between facets along all the axis (True
            for backward overlap, False for forward overlap). By default False.

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
        if not image_size.size == data_size.size:
            raise ValueError(
                "image_size and data_size must have the same number of elements"
            )

        super(SyncLinearConvolution, self).__init__(image_size, data_size)
        self.grid_size = np.array(grid_size, dtype="i")
        self.comm = comm

        # * Cartesian communicator and nd rank
        self.cartcomm = self.comm.Create_cart(
            dims=grid_size,
            periods=self.ndims * [circular_boundaries],
            reorder=False,
        )
        self.rank = comm.Get_rank()
        # self.ranknd = np.unravel_index(self.rank, grid_size)
        self.ranknd = np.array(self.cartcomm.Get_coords(self.rank), dtype="i")

        # * useful dimensions
        if not len(kernel.shape) == self.ndims:
            raise ValueError("kernel should have ndims = len(image_size) dimensions")
        if kernel.dtype.kind == "c":
            raise TypeError("only real-valued kernel supported")
        self.overlap_size = np.array(kernel.shape, dtype="i") - 1
        tile_pixels = ucomm.local_split_range_nd(
            self.grid_size, self.image_size, self.ranknd, backward=backward
        )
        self.tile_size = tile_pixels[:, 1] - tile_pixels[:, 0] + 1
        (
            self.local_data_size,
            self.facet_size,
            self.facet_size_adj,
        ) = calculate_local_data_size(
            self.tile_size,
            self.ranknd,
            self.overlap_size,
            self.grid_size,
            backward=backward,
        )

        # facet (convolution)
        self.local_conv_size = self.facet_size + self.overlap_size
        self.offset = self.facet_size - self.tile_size
        self.offset_adj = self.facet_size_adj - self.tile_size

        self.fft_kernel = np.fft.rfftn(kernel, self.local_conv_size)

        # * useful slices
        # extract valid coefficients after local convolutions
        self.local_slice_valid_conv = ucomm.slice_valid_coefficients(
            self.ranknd, self.grid_size, self.overlap_size
        )
        # retrieve image tile from fft-based convolution (adjoint conv. operator)
        self.local_slice_conv = tuple(
            [np.s_[: self.tile_size[d]] for d in range(self.ndims)]
        )
        # slice to set value of local convolution in the adjoint buffer
        self.local_slice_conv_adj = ucomm.get_local_slice(
            self.ranknd, self.grid_size, self.offset_adj, backward=not backward
        )

        # * implementation of the distributed direct operator
        self.direct_communicator = comms.SyncCartesianCommunicator(
            self.comm,
            self.cartcomm,
            grid_size,
            itemsize,
            self.facet_size,
            self.overlap_size,
            backward,
        )

        # * implementation of the distributed adjoint operator
        self.adjoint_communicator = comms.SyncCartesianCommunicator(
            self.comm,
            self.cartcomm,
            grid_size,
            itemsize,
            self.facet_size_adj,
            self.overlap_size,
            not backward,
        )

    # ? change interface to pass the output array, to be updated in place?
    # (would need to do the same for the serial case)
    def forward(self, input_image):
        r"""Implementation of the direct operator to update the input array
        ``input_image`` (from image to data space).

        Parameters
        ----------
        input_image : numpy.ndarray of float
            Input buffer array (image space), of size ``self.facet_size``.

        Returns
        -------
        y : numpy.ndarray
            Result of the direct operator using the information from the local
            image facet.

        Note
        ----
        The input buffer ``input_image`` is updated in-place.
        """
        self.direct_communicator.update_borders(input_image)
        y = fft_conv(input_image, self.fft_kernel, self.local_conv_size)[
            self.local_slice_valid_conv
        ]
        return y

    # ? change interface to pass the output array, to be updated in place?
    # (would need to do the same for the serial case)
    def adjoint(self, input_data):
        r"""Implementation of the adjoint operator to update the input array
        ``input_data`` (from data to image space).

        Parameters
        ----------
        input_data : numpy.ndarray of float
            Input buffer array (data space), of size ``self.facet_size_adj``.

        Returns
        -------
        x : numpy.ndarray
            Result of the adjoint operator using the information from the local
            data facet.

        Note
        ----
        The input buffer ``input_data`` is updated in-place.
        """
        self.adjoint_communicator.update_borders(input_data)
        x = fft_conv(input_data, np.conj(self.fft_kernel), self.local_conv_size)[
            self.local_slice_conv
        ]
        return x
