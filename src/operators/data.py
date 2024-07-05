"""Helper functions to generate, save and load the synthetic data using several
MPI processes.
"""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# reference: P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.

# TODO: to be simplified, e.g. with checkpoint and model objects, or even
# TODO: removed

from os.path import splitext

import h5py
import numpy as np
from mpi4py import MPI
from PIL import Image
from scipy.signal.windows import gaussian

import dsgs.operators.fessler_convolution as fconv
import dsgs.utils.communications as ucomm
from dsgs.functionals.numpy.prox import prox_nonegativity
from dsgs.operators.convolutions import fft_conv
from dsgs.operators.distributed_convolutions import create_local_to_global_slice
from dsgs.operators.linear_convolution import SerialConvolution


def get_image(imagefilename, M=None):
    r"""Load an image from a .png of .h5 file, ensuring all pixel values are
    nonnegative.

    Parameters
    ----------
    imagefilename : string
        Full path and name of the image to be loaded (including extension).
    M : double, optional
        Maximum intensity value selected, by default None.

    Returns
    -------
    numpy.ndarray
        Loaded image (double format entries).
    """
    ext = splitext(imagefilename)[-1]

    if ext == ".h5":
        with h5py.File(imagefilename, "r") as f:
            x = f["x"][()]
    else:  # .png file by default
        # x = mpimg.imread(imagefilename).astype(np.double)
        img = Image.open(imagefilename, "r")
        x = np.asarray(img).astype(np.double)

    # ! make sure no pixel is 0 or lower
    if M is None:
        M = np.max(x)
    x[x <= 0] = np.min(x[x > 0])  # np.finfo(x.dtype).eps
    x = M * x / np.max(x)
    prox_nonegativity(x)

    return x


def generate_2d_gaussian_kernel(kernel_size, kernel_std):
    r"""Generate a square normalized 2D Gaussian kernel.

    Parameters
    ----------
    kernel_size : int
        Size of one dimension of the kernel.
    kernel_std : float
        Standard deviation of the Gaussian kernel.

    Note
    ----
    Equivalent to the ``fspecial('gaussian', ...)`` function in Matlab.

    Returns
    -------
    h : numpy.ndarray
        Square Gaussian kernel with :math:`\|h\|_1 = 1`.
    """
    # equivalent to fspecial('gaussian', ...) in Matlab
    w = gaussian(kernel_size, kernel_std)
    h = w[:, np.newaxis] * w[np.newaxis, :]
    h = h / np.sum(h)
    return h


def generate_random_mask(image_size, percent, rng):
    r"""Generate a random inpainting mask.

    Parameters
    ----------
    image_size : numpy.ndarray of int
        Shape of the image to be masked.
    percent : float
        Fraction of masked pixels, ``1-p`` corresponding to the fraction of
        observed pixels in the image (``p`` needs to be nonnegative, smaller
        than or equal to 1).
    rng : numpy.random.Generator
        Numpy random number generator.

    Returns
    -------
    mask : numpy.ndarray of bool
        Masking operator such that ``mask.shape == image_size``.

    Raises
    ------
    ValueError
        Fraction of observed pixels ``percent`` should be such that
        :math:`0 \leq p \leq 1`.
    """
    if percent < 0 or percent > 1:
        raise ValueError(
            "Fraction of observed pixels percent should be such that: 0 <= percent <= 1."
        )

    N = np.prod(image_size)  # total number of pixels
    masked_id = np.unravel_index(
        rng.choice(N, (percent * N).astype(int), replace=False), image_size
    )
    mask = np.full(image_size, True, dtype=bool)
    mask[masked_id] = False

    return mask


def generate_local_poisson_inpainting_data(
    comm, cartcomm, grid_size, ranknd, tile, local_mask, tile_pixels
):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    comm : _type_
        _description_
    cartcomm : _type_
        _description_
    grid_size : _type_
        _description_
    ranknd : _type_
        _description_
    tile : _type_
        _description_
    local_mask : _type_
        _description_
    tile_pixels : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # * communicator info
    size = comm.Get_size()
    rank = comm.Get_rank()

    # * parallel rng
    child_seed = None
    if rank == 0:
        ss = np.random.SeedSequence(1234)
        child_seed = ss.spawn(size)
    local_seed = comm.scatter(child_seed, root=0)
    local_rng = np.random.default_rng(local_seed)

    # * tile size and indices in the global image / data
    tile_size = tile_pixels[:, 1] - tile_pixels[:, 0] + 1
    local_data_size = tile_size

    # * generate masked data
    Hx = np.zeros(tile_size, dtype="d")
    Hx[local_mask] = tile[local_mask]
    local_data = np.zeros(tile_size, dtype="d")
    local_data[local_mask] = local_rng.poisson(Hx[local_mask])

    # * compute number of observations across all the workers
    global_data_size = np.zeros(1, dtype="i")
    local_size = np.sum(local_mask, dtype="i")
    comm.Allreduce(local_size, global_data_size, op=MPI.SUM)

    # slice for indexing into global arrays
    global_slice_tile = create_local_to_global_slice(
        tile_pixels,
        ranknd,
        np.zeros(tile_size.size, dtype="i"),
        local_data_size,
    )
    return local_data, Hx, global_slice_tile, global_data_size


def generate_local_gaussian_inpainting_data(
    comm, cartcomm, grid_size, ranknd, tile, local_mask, tile_pixels, isnr
):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    comm : _type_
        _description_
    cartcomm : _type_
        _description_
    grid_size : _type_
        _description_
    ranknd : _type_
        _description_
    tile : _type_
        _description_
    local_mask : _type_
        _description_
    tile_pixels : _type_
        _description_
    isnr : bool
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # * communicator info
    size = comm.Get_size()
    rank = comm.Get_rank()

    # * parallel rng
    child_seed = None
    if rank == 0:
        ss = np.random.SeedSequence(1234)
        child_seed = ss.spawn(size)
    local_seed = comm.scatter(child_seed, root=0)
    local_rng = np.random.default_rng(local_seed)

    # * tile size and indices in the global image / data
    tile_size = tile_pixels[:, 1] - tile_pixels[:, 0] + 1
    local_data_size = tile_size

    # * generate masked data
    Hx = np.zeros(tile_size, dtype="d")
    Hx[local_mask] = tile[local_mask]

    # * compute noise variance across all workers
    sqnorm_Hx = np.zeros(1, dtype="d")
    local_sqnorm_Hx = np.sum(Hx[local_mask] ** 2)
    global_data_size = np.zeros(1, dtype="i")
    local_size = np.sum(local_mask, dtype="i")

    comm.Allreduce(local_sqnorm_Hx, sqnorm_Hx, op=MPI.SUM)
    comm.Allreduce(local_size, global_data_size, op=MPI.SUM)
    sig2 = 10 ** (-isnr / 10) * sqnorm_Hx / global_data_size

    # * generate noisy data
    local_data = np.zeros(tile_size, dtype="d")
    local_data[local_mask] = Hx[local_mask] + np.sqrt(sig2) * local_rng.standard_normal(
        size=local_size
    )

    # * slice for indexing into global arrays
    global_slice_tile = create_local_to_global_slice(
        tile_pixels,
        ranknd,
        np.zeros(tile_size.size, dtype="i"),
        local_data_size,
    )
    return local_data, Hx, global_slice_tile, global_data_size, sig2


def mpi_load_image_from_h5(comm, grid_size, ranknd, filename, image_size, overlap_size):
    ndims = image_size.size

    # * useful sizes
    tile_pixels = ucomm.local_split_range_nd(grid_size, image_size, ranknd)
    tile_size = tile_pixels[:, 1] - tile_pixels[:, 0] + 1
    facet_pixels = ucomm.local_split_range_nd(
        grid_size, image_size, ranknd, overlap=overlap_size, backward=False
    )
    facet_size = facet_pixels[:, 1] - facet_pixels[:, 0] + 1
    offset = facet_size - tile_size

    print(
        "Process {}: facet_size={}, offset={}".format(
            comm.Get_rank(), facet_size, offset
        )
    )

    # * setup useful slices
    # forward overlap
    # local_slice_tile = tuple([np.s_[: tile_size[d]] for d in range(ndims)])
    local_slice_tile = ucomm.get_local_slice(ranknd, grid_size, offset, backward=False)

    global_slice_tile = tuple(
        [np.s_[tile_pixels[d, 0] : tile_pixels[d, 1] + 1] for d in range(ndims)]
    )

    print(
        "Process {}: local_slice_tile={}, offset={}".format(
            comm.Get_rank(), local_slice_tile, offset
        )
    )

    # * parallel loading
    f = h5py.File(filename, "r", driver="mpio", comm=comm)
    dset = f["x"]
    facet = np.empty(facet_size, dtype="d")
    dset.read_direct(
        facet,
        global_slice_tile,
        local_slice_tile,
    )
    f.close()

    return facet, tile_pixels


def mpi_write_image_to_h5(
    comm, grid_size, ranknd, filename, image_size, local_slice_tile, facet, var_name
):
    ndims = image_size.size

    # * useful sizes
    tile_pixels = ucomm.local_split_range_nd(grid_size, image_size, ranknd)

    # * setup useful slices
    global_slice_tile = tuple(
        [np.s_[tile_pixels[d, 0] : tile_pixels[d, 1] + 1] for d in range(ndims)]
    )

    # * parallel write
    f = h5py.File(filename, "w", driver="mpio", comm=comm)
    dset = f.create_dataset(var_name, image_size, dtype="d")
    dset.write_direct(
        facet,
        local_slice_tile,
        global_slice_tile,
    )
    f.close()

    pass


def generate_data(x, h):
    # serial version, linear convolution

    # * rng
    rng = np.random.default_rng(1234)

    # local convolution
    data_size = np.array(x.shape, dtype="i") + np.array(h.shape, dtype="i") - 1
    fft_h = np.fft.rfftn(h, data_size)

    # ! issue: need to make sure Hx >= 0, not necessarily the case numerically
    # ! with a fft-based convolution
    # https://github.com/pytorch/pytorch/issues/30934
    # Hx = scipy.ndimage.convolve(x, h, output=Hx, mode='constant', cval=0.0)
    # Hx = convolve2d(x, h, mode='full')
    Hx = fft_conv(x, fft_h, data_size)
    prox_nonegativity(Hx)
    data = rng.poisson(Hx).astype(np.double)

    return data, Hx


def generate_data_gaussian_deconvolution(x, h, isnr, cconv=False):
    # serial version, linear convolution

    # * rng
    rng = np.random.default_rng(1234)
    N = np.array(x.shape, dtype="i")

    # local convolution
    if cconv:
        data_size = np.array(x.shape, dtype="i")
    else:
        data_size = np.array(x.shape, dtype="i") + np.array(h.shape, dtype="i") - 1

    # * Debug: revise data generation (rely on same object as inference)
    # fft_h = np.fft.rfftn(h, data_size)
    # Hx = fft_conv(x, fft_h, data_size)

    model = SerialConvolution(N, h, data_size)
    Hx = model.forward(x)

    # # TODO: to be modified to investigate artefacts? (difference in the
    # # TODO: - convolution models between inference and data generation?)
    # sig2 = 10 ** (-isnr / 10) * np.sum(Hx**2) / Hx.size
    # # ! adopt the same def. for the noise level as the one used for the
    # # ! - code published for :cite:p:`Vono2019tsp`
    sig2 = 10 ** (-isnr / 10) * np.var(Hx)

    # # prox_nonegativity(Hx)
    data = Hx + np.sqrt(sig2) * rng.standard_normal(size=Hx.shape)

    return data, Hx, sig2


def generate_data_gaussian_deconvolution2(x, h, isnr):
    # serial version, linear convolution

    # * rng
    rng = np.random.default_rng(1234)
    # N = np.array(x.shape, dtype="i")

    # local convolution
    data_size = np.array(x.shape, dtype="i") + np.array(h.shape, dtype="i") - 1

    # * Debug: revise data generation (rely on same object as inference)
    # fft_h = np.fft.rfftn(h, data_size)
    # Hx = fft_conv(x, fft_h, data_size)

    model = fconv.SerialConvolution(h, data_size)
    Hx = model.forward(x)

    # ! adopt the same def. for the noise level as the one used for the
    # ! - code published for :cite:p:`Vono2019tsp`
    sig2 = 10 ** (-isnr / 10) * np.var(Hx)

    # # prox_nonegativity(Hx)
    data = Hx + np.sqrt(sig2) * rng.standard_normal(size=Hx.shape)

    return data, Hx, sig2


def generate_local_data(
    comm, cartcomm, grid_size, ranknd, facet, h, image_size, tile_pixels, backward=False
):
    # distributed version, linear convolution

    # * communicator info
    size = comm.Get_size()
    rank = comm.Get_rank()
    ndims = image_size.size
    kernel_size = np.array(h.shape, dtype="i")
    overlap_size = kernel_size - 1

    # * parallel rng
    child_seed = None
    if rank == 0:
        ss = np.random.SeedSequence(1234)
        child_seed = ss.spawn(size)
    local_seed = comm.scatter(child_seed, root=0)
    local_rng = np.random.default_rng(local_seed)

    # * facet / tile size and indices in the global image / data
    # tile_pixels = ucomm.local_split_range_nd(grid_size, image_size, ranknd)
    # facet_pixels = ucomm.local_split_range_nd(
    #     grid_size, image_size, ranknd, overlap=overlap_size
    # )
    # facet_size = facet_pixels[:, 1] - facet_pixels[:, 0] + 1
    tile_size = tile_pixels[:, 1] - tile_pixels[:, 0] + 1
    facet_size = np.array(facet.shape, dtype="i")

    # backward overlap
    # ! larger number of data points on the border
    # local_data_size = (
    #     tile_size + (ranknd == grid_size - 1) * overlap_size
    # )
    # forward overlap
    local_data_size = tile_size + (ranknd == 0) * overlap_size
    local_conv_size = facet_size + overlap_size

    # * communications to compute the data (distributed convolution)
    # facet = np.empty(facet_size, dtype="d")
    (
        dest,
        src,
        resizedsendsubarray,
        resizedrecvsubarray,
    ) = ucomm.setup_border_update(
        cartcomm, ndims, facet.itemsize, facet_size, overlap_size, backward=False
    )

    for d in range(ndims):
        comm.Sendrecv(
            [facet, 1, resizedsendsubarray[d]],
            dest[d],
            recvbuf=[facet, 1, resizedrecvsubarray[d]],
            source=src[d],
        )

    # * free custom types
    # for d in range(ndims):
    #     if isvalid_comm[d]:
    #         resizedsendsubarray[d].Free()
    #         resizedrecvsubarray[d].Free()

    # local convolution
    fft_h = np.fft.rfftn(h, local_conv_size)
    # H = np.fft.rfft2(
    #     np.fft.fftshift(padding.pad_array(h, N, padmode="around"))
    # )  # doing as in Vono's reference code
    local_coeffs = ucomm.slice_valid_coefficients(ranknd, grid_size, overlap_size)

    # ! issue: need to make sure Hx >= 0, not necessarily the case numerically
    # ! with a fft-based convolution
    # https://github.com/pytorch/pytorch/issues/30934
    # Hx = scipy.ndimage.convolve(facet, h, output=Hx, mode='constant', cval=0.0)
    # Hx = convolve2d(facet, h, mode='full')[local_coeffs]
    Hx = fft_conv(facet, fft_h, local_conv_size)[local_coeffs]
    prox_nonegativity(Hx)
    local_data = local_rng.poisson(Hx)

    # slice for indexing into global arrays
    global_slice_data = create_local_to_global_slice(
        tile_pixels,
        ranknd,
        overlap_size,
        local_data_size,
        backward=backward,
    )

    return local_data, Hx, global_slice_data


def mpi_save_data_to_h5(
    comm, filename, data_size, local_data, local_clean_data, global_slice_data
):
    f = h5py.File(filename, "r+", driver="mpio", comm=comm)
    dset = f.create_dataset("data", data_size, dtype="d")
    dset[global_slice_data] = local_data

    dset = f.create_dataset("clean_data", data_size, dtype="d")
    dset[global_slice_data] = local_clean_data
    f.close()

    return


def mpi_load_data_from_h5(
    comm,
    ranknd,
    filename,
    data_size,
    local_data_size,
    tile_pixels,
    overlap_size,
    var_name="data",
    backward=True,
):
    ndims = data_size.size

    # slice for indexing into global arrays
    # global_slice_data = tuple(
    #     [
    #         np.s_[tile_pixels[d, 0] : tile_pixels[d, 0] + local_data_size[d]]
    #         for d in range(ndims)
    #     ]
    # )

    local_slice = tuple(ndims * [np.s_[:]])
    global_slice_data = create_local_to_global_slice(
        tile_pixels, ranknd, overlap_size, local_data_size, backward=backward
    )

    # loading data
    f = h5py.File(filename, "r", driver="mpio", comm=comm)
    dset = f[var_name]
    local_data = np.empty(local_data_size, dtype="d")
    dset.read_direct(
        local_data,
        global_slice_data,
        local_slice,
    )
    f.close()

    return local_data


# if __name__ == "__main__":
#     pass
