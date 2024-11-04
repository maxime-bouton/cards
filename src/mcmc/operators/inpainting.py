import numpy as np
from mpi4py import MPI

import mcmc.utils.communications as ucomm  #! deleted folder
from mcmc.operators.linear_operator import LinearOperator


# * Inpainting model class (serial and distributed)
class SerialInpainting(LinearOperator):
    r"""Model class implementing an inpainting operator in a serial algorithm.

    Attributes
    ----------
    image_size : numpy.ndarray of int, of size ``d``
        Full image size.
    mask : numpy.ndarray
        Input mask operator. The array should have ``d`` axis, such that
        ``mask.shape[i] == image_size[i]`` for ``i in range(d)``.
    mask_id : numpy.ndarray of int
        Array of indices corresponding to the observed points (as given by
        ``numpy.nonzero``).
    """

    def __init__(
        self,
        image_size,
        mask,
    ):
        r"""SerialInpainting constructor.

        Parameters
        ----------
        image_size : numpy.ndarray of int, of size ``d``
            Full image size.
        mask : numpy.ndarray
            Input mask operator for the current process. The array should have
            ``d`` axis, such that ``mask.shape[i] == image_size[i]`` for
            ``i in range(d)``.

        Raises
        ------
        ValueError
            ``mask`` and image should have the same size.
        """
        if not np.all(mask.shape == image_size):
            raise ValueError("mask and image should have the same size")

        # ! flattened data dimension after applying the inpainting (= cropping)
        # ! operator
        data_size = np.array([np.sum(mask)], dtype="i")
        super(SerialInpainting, self).__init__(image_size, data_size)

        self.mask = mask
        self.mask_id = np.nonzero(mask)

    def forward(self, input_image):
        r"""Implementation of the direct operator to update the input array
        ``input_image`` (from image to data space, cropping).

        Parameters
        ----------
        input_image : numpy.ndarray of float
            Input array (image space).

        Returns
        -------
        numpy.ndarray
            Result of the inpainting operator (direct operator).
        """
        # self.mask * input_image
        return input_image[self.mask_id]

    def adjoint(self, input_data):
        r"""Implementation of the adjoint operator to update the input array
        ``input_data`` (from data to image space, zero-padding).

        Parameters
        ----------
        input_data : numpy.ndarray of float
            Input array (data space).

        Returns
        -------
        numpy.ndarray
            Result of the inpainting operator (inpainting operator is auto-adjoint).
        """
        # self.mask * input_data
        output_image = np.zeros(self.image_size, dtype=input_data.dtype)
        output_image[self.mask_id] = input_data
        return output_image


class SyncInpainting(LinearOperator):
    r"""Synchronous distributed implementation of an inpainting operator. Once
    distributed, MPI processes containing different image tiles never need to
    communicate.

    Attributes
    ----------
    image_size : numpy.ndarray of int, of size ``d``
        Full image size.
    mask : numpy.ndarray
        Input chunk of a mask operator fir the current MPI process. The
        overall array should have ``d`` axis.
    grid_size : list of int, of size ``d``
        Number of workers along each of the ``d`` dimensions of the
        communicator grid.
    data_size : numpy.ndarray of int
        Total number of observed data points (over all the MPI processes).
    local_data_size : int
        Number of observed data points defined from the local ``mask``.
    mask_id : numpy.ndarray of int
        Array of indices corresponding to the local observed points (as given
        by ``numpy.nonzero``).
    comm : mpi4py.MPI.Comm
        MPI communicator unferlying all the communications.
    cartcomm : mpi4py.MPI.Cartcomm
        Cartesian MPI communicator underlying the communications.
    ndims : int
        Number of dimensions (axis) of the model.
    rank : int
        Rank of the current MPI-process.
    ranknd : numpy.ndarray[int]
        Multi-linear rank of the current MPI-process in the Cartesian grid of
        workers (nD setting).
    tile_size : numpy.ndarray[int]
        Size of the non-overlapping tiles handled by each MPI process for the
        direct convolution operator.
    """

    def __init__(
        self,
        image_size,
        mask,
        comm,
        grid_size,
    ):
        r"""Synchronous distributed implementation of an inpainting operator.

        Parameters
        ----------
        image_size : numpy.ndarray of int, of size ``d``
            Full image size.
        mask : numpy.ndarray
            Input chunk of a mask operator fir the current MPI process. The
            overall array should have ``d`` axis.
        comm : mpi4py.MPI.Comm
            Underlying MPI communicator.
        grid_size : list of int, of size ``d``
            Number of workers along each of the ``d`` dimensions of the
            communicator grid.

        Raises
        ------
        ValueError
            local mask and image should have the same size.
        """
        # ! flattened data dimension after applying the inpainting (= cropping)
        # ! operator
        self.local_data_size = np.array(np.sum(mask), dtype="i")
        data_size = np.empty(1, dtype="i")
        comm.Reduce(
            [self.local_data_size, MPI.INT],
            [data_size, MPI.INT],
            op=MPI.SUM,
        )

        super(SyncInpainting, self).__init__(image_size, data_size)
        self.grid_size = np.array(grid_size, dtype="i")
        self.comm = comm

        # * Cartesian communicator and nd rank
        self.cartcomm = self.comm.Create_cart(
            dims=grid_size,
            periods=self.ndims * [False],
            reorder=False,
        )
        self.rank = comm.Get_rank()
        # self.ranknd = np.unravel_index(self.rank, grid_size)
        self.ranknd = np.array(self.cartcomm.Get_coords(self.rank), dtype="i")

        # * useful dimensions
        tile_pixels = ucomm.local_split_range_nd(
            self.grid_size, self.image_size, self.ranknd
        )
        self.tile_size = tile_pixels[:, 1] - tile_pixels[:, 0] + 1

        if not np.all(mask.shape == self.tile_size):
            raise ValueError("local mask and image tile should have the same size")

        self.mask = mask
        self.mask_id = np.nonzero(mask)

    # ? change interface to pass the output array, to be updated in place?
    # ? add a new function for in-place update?
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
        """
        # y = self.mask * input_image
        y = input_image[self.mask_id]
        return y

    # ? change interface to pass the output array, to be updated in place?
    # ? add a new function for in-place update?
    def adjoint(self, input_data):
        r"""Implementation of the adjoint operator to update the input array
        ``input_data`` (from data to image space).

        Parameters
        ----------
        input_data : numpy.ndarray of float
            Input buffer array (data space), of size ``self.data_size``.

        Returns
        -------
        x : numpy.ndarray
            Result of the adjoint operator using the information from the local
            data facet.
        """
        # x = self.mask * input_data
        x = np.zeros(self.tile_size, dtype=input_data.dtype)
        x[self.mask_id] = input_data
        return x
