""" Distributed circular convolution operator. Includes helper functions to
implement the communications leveraged for the distributed implementation
of the circular convolution and its adjoint.
"""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# reference: P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.

import numpy as np
from mpi4py import MPI

import dsgs.utils.communications as ucomm
import dsgs.utils.communicators as comms
from dsgs.operators.convolutions import fft_conv
from dsgs.operators.linear_operator import LinearOperator
from dsgs.operators.mpi_adjoint_circular_padding import AdjointCircularPadding
from dsgs.operators.padding import pad_array_nd


def calculate_local_data_size_cconv(
    tile_size, ranknd, overlap_size, grid_size, backward=True
):
    r"""Compute the size of the chunk convolution hold by the
    current worker.

    Parameters
    ----------
    tile_size : numpy.ndarray[int]
        Size of the non-overlapping image tile underlying the overlapping
        facets.
    ranknd : numpy.ndarray[int]
        Rank of the current process in the nD grid of MPI processes.
    overlap_size : numpy.ndarray[int]
        Size of the overlap along each dimension.
    grid_size : numpy.ndarray[int]
        Number of processes along each dimension of the nD MPI process grid.
    backward : bool, optional
        Direction of the overlap in the cartesian grid along all the
        dimensions (forward or backward overlap), by default True.

    Returns
    -------
    local_data_size : numpy.ndarray[int]
        Size of the local chunk of the data owned by the current process.
    facet_size : numpy.ndarray[int]
        Size of the overlapping facet handled by the current process (direct
        operator).
    facet_size_adj : numpy.ndarray[int]
        Size of the overlapping facet handled by the current process (adjoint
        operator).
    """
    # ! 0-padding handled separately at another level
    # ! for communication, need local size of buffer after convolution
    if backward:
        local_data_size = tile_size
        facet_size = tile_size + overlap_size
        facet_size_adj = local_data_size + (ranknd < grid_size - 1) * overlap_size
    else:
        local_data_size = tile_size
        facet_size = tile_size + overlap_size
        facet_size_adj = local_data_size + (ranknd > 0) * overlap_size

    return local_data_size, facet_size, facet_size_adj


def slice_valid_coefficients_cconv(ranknd, grid_size, overlap_size):
    r"""Helper elements to extract the valid local convolution coefficients
    when computing a direct circular convolution using linear convolution
    operators.

    Returns slice to select the valid local convolution coefficients, with the
    necessary padding parameters to implement the adjoint operator.

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
    valid_conv_coefficients : tuple[slice]
        Slice to extract valid coefficients from the local convolutions.

    Raises
    ------
    AssertionError
        `ranknd`, `grid_size` and `overlap_size` must all have the save shape.

    Remark
    ------
    The circular operator can be decomposed as follows:
    - pre-padding the input signal using circular boundary condition;
    - computing the linear convolution with the convolution kernel;
    - cropping the convolution output on both sides.
    """

    ndims = ranknd.size

    if not (grid_size.size == ndims and overlap_size.size == ndims):
        raise AssertionError(
            r"`ranknd`, `grid_size` and `overlap_size` must have the save \
                shape"
        )

    L = ndims * [None]
    R = ndims * [None]

    # ! cropping on both side for the direct operator
    for d in range(ndims):
        L[d] = overlap_size[d]
        R[d] = -overlap_size[d]

    valid_conv_coefficients = tuple([np.s_[L[d] : R[d]] for d in range(ndims)])

    return valid_conv_coefficients


def slice_valid_coefficients_cconv_adjoint(
    ranknd, grid_size, overlap_size, backward=False
):
    r"""Helper elements to extract the valid local convolution coefficients
    when computing the adjoint circular convolution using linear convolution
    operators.

    Returns slice to select the valid local convolution coefficients, with the
    necessary padding parameters to implement the adjoint operator.

    Parameters
    ----------
    ranknd : numpy.ndarray[int]
        Rank of the process in a Cartesian nD grid of MPI processes.
    grid_size : numpy.ndarray[int]
        Size of the MPI process grid.
    overlap_size : numpy.ndarray[int]
        Overlap between contiguous facets along each dimension.
    backward : bool, optional
        Orientation of the overlap along the dimensions, by default True.

    Returns
    -------
    valid_conv_coefficients : tuple[slice]
        Slice to extract valid coefficients from the local convolutions.

    Raises
    ------
    AssertionError
        `ranknd`, `grid_size` and `overlap_size` must all have the save shape.

    Remark
    ------
    For some cores, the local convolutions are performed with slightly more
    padding than necessary to avoid using a smaller discrete Fourier
    transform for the convolution kernel compared to the direct operator. A
    larger cropping is thus applied in such cases.
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
        R[d] = -2 * overlap_size[d]
        if backward and ranknd[d] == 0:
            R[d] = -overlap_size[d]
        if (not backward) and ranknd[d] == grid_size[d] - 1:
            R[d] = -overlap_size[d]

    valid_conv_coefficients = tuple([np.s_[L[d] : R[d]] for d in range(ndims)])

    return valid_conv_coefficients


def get_local_slice_cconv(ranknd, grid_size, overlap_size, backward=True):
    r"""Slice to extract the pixels specific to a given worker.

    Get the slice corresponding to the elements exclusively handled by the
    current process (i.e., remove the overlap from overlapping facets).

    Parameters
    ----------
    ranknd : numpy.ndarray[int]
        Rank of the current process in the nD Cartesian grid of MPI processes.
    grid_size : numpy.ndarray[int]
        Size of the process grid
    overlap_size : numpy.ndarray[int]
        Size of the overlap between contiguous facets.
    backward : bool, optional
        Orientation of the overlap along the dimensions, by default True.
    adjoint : bool, optional
        Indicates whether the selection is considered for the adjoint or the
        direct operator, by default False.

    Returns
    -------
    slice_facet : tuple[slice]
        Slice to extract the coefficients in image space specifically handled
        by the current process.
    slice_adjoint_facet : tuple[slice]
        Slice to extract the coefficients in data space specifically handled by
        the current process.

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

    local_slice = ndims * [np.s_[:]]
    local_slice_adjoint = ndims * [np.s_[:]]
    isvalid_splitting = overlap_size > 0

    if backward:
        for d in range(ndims):
            if isvalid_splitting[d]:
                local_slice[d] = np.s_[overlap_size[d] :]
                if ranknd[d] < grid_size[d] - 1:
                    local_slice_adjoint[d] = np.s_[: -overlap_size[d]]
    else:
        for d in range(ndims):
            if isvalid_splitting[d]:
                local_slice[d] = np.s_[: -overlap_size[d]]
            if ranknd[d] > 0:
                local_slice_adjoint[d] = np.s_[overlap_size[d] :]

    return tuple(local_slice), tuple(local_slice_adjoint)


# * Circular convolution model class (distributed)
class SyncCircularConvolution(LinearOperator):
    r"""Synchronous MPI-distributed implementation of a circular convolution
    operator.

    Attributes
    ----------
    image_size : numpy.ndarray of int, of size ``d``
        Full image size.
    kernel : numpy.ndarray of float
        Input convolution kernel.
    grid_size : list of int, of size ``d``
        Number of workers along each of the ``d`` dimensions of the
        communicator grid.
    itemsize : numpy.dtype.itemsize
        Size (in bytes) of the scalar type to be handled during the
        communications.
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
        Slice object to extract valid coefficients after local convolutions (
        direct operator).
    local_slice_valid_conv_adj
        Slice object to extract valid coefficients after local convolutions (
        adjoint operator).
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
        kernel,
        comm,
        grid_size,
        itemsize,
        backward=False,
    ):
        r"""Synchronous distributed implementation of a (linear) convolution
        model.

        Parameters
        ----------
        image_size : numpy.ndarray of int, of size ``d``
            Full image size.
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
        backward : bool, optional
            Direction of the overlap between facets along all the axis (True
            for backward overlap, False for forward overlap). By default False.

        Raises
        ------
        ValueError
            ``kernel`` should have ``ndims = len(image_size)`` dimensions.
        TypeError
            Only real-valued kernel supported.
        """
        super(SyncCircularConvolution, self).__init__(image_size, image_size)
        self.grid_size = np.array(grid_size, dtype="i")
        self.comm = comm

        # * Cartesian communicator and nd rank
        self.cartcomm = self.comm.Create_cart(
            dims=grid_size,
            periods=self.ndims * [True],
            reorder=False,
        )
        self.cartcomm_adjoint = self.comm.Create_cart(
            dims=grid_size,
            periods=self.ndims * [False],
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
        ) = calculate_local_data_size_cconv(
            self.tile_size,
            self.ranknd,
            self.overlap_size,
            self.grid_size,
            backward=backward,
        )

        # facet (convolution)
        # TODO: check if conv size is correct for all workers
        self.local_conv_size = self.facet_size + self.overlap_size
        # self.local_conv_size = self.tile_size + 2*self.overlap_size
        self.offset = self.facet_size - self.tile_size
        self.offset_adj = self.facet_size_adj - self.tile_size

        self.fft_kernel = np.fft.rfftn(kernel, self.local_conv_size)

        # * useful slices
        # extract valid coefficients after local convolutions
        self.local_slice_valid_conv = slice_valid_coefficients_cconv(
            self.ranknd, self.grid_size, self.overlap_size
        )
        self.local_slice_valid_conv_adj = slice_valid_coefficients_cconv_adjoint(
            self.ranknd, self.grid_size, self.overlap_size, backward=backward
        )

        # slice to set value in the buffer (direct and adjoint operator)
        self.local_slice_conv, self.local_slice_conv_adj = get_local_slice_cconv(
            self.ranknd, self.grid_size, self.overlap_size, backward=backward
        )

        # * communications for the distributed direct operator
        self.direct_communicator = comms.SyncCartesianCommunicator(
            self.comm,
            self.cartcomm,
            grid_size,
            itemsize,
            self.facet_size,
            self.overlap_size,
            backward=backward,
        )

        self.adjoint_communicator = comms.SyncCartesianCommunicator(
            self.comm,
            self.cartcomm_adjoint,
            grid_size,
            itemsize,
            self.facet_size_adj,
            self.overlap_size,
            backward=not backward,
        )

        # * adjoint circular padding operator
        self.adjoint_padding = AdjointCircularPadding(
            self.comm,
            self.cartcomm_adjoint,
            self.grid_size,
            itemsize,
            self.tile_size,
            self.overlap_size,
            backward=backward,
        )

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
        # ! distributed circshift is needed for forward overlap, to be
        # ! investigated
        return y

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
        # TODO: distributed circshift would be needed for forward overlap
        self.adjoint_communicator.update_borders(input_data)

        # ! 0-padding (before or after) at the boundaries, before convolutions
        pad_width = [
            [
                self.overlap_size[d] * (self.ranknd[d] == 0),
                self.overlap_size[d] * (self.ranknd[d] == self.grid_size[d] - 1),
            ]
            for d in range(self.ndims)
        ]
        y = np.pad(input_data, pad_width, mode="constant")

        # print("Process {}: y.shape ={}".format(self.adjoint_communicator.rank, y.shape))

        # ! fft_conv remove C-contiguous flag from the array
        x = np.ascontiguousarray(
            fft_conv(y, np.conj(self.fft_kernel), self.local_conv_size)[
                self.local_slice_valid_conv_adj
            ]
        )
        # TODO: revise cropping for the dimension along which there is no
        # TODO- splitting

        # print(
        #     "Before crop: process {}: x.shape={}, x.flags ={}".format(
        #         self.adjoint_communicator.rank, x.shape, x.flags
        #     )
        # )

        # * adjoint circular convolution
        z = self.adjoint_padding.update_borders(x)

        # print(
        #     "After crop: process {}: z.shape={}".format(
        #         self.adjoint_communicator.rank, z.shape
        #     )
        # )

        return z
