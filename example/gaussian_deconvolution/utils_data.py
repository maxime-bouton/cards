import json

import h5py
import numpy as np
import scipy

from mcmc.utils.utils import apply_gaussian_noise, generate_observations, load_img_size


def load_sizes_from_h5(filename):
    with h5py.File(filename, "r") as file:
        return file["data"].shape, file["kernel"].shape


def load_from_h5(filename, local_slice=slice(None)):
    """load the mask01, sig2 and data entries from the h5 file"""
    with h5py.File(filename, "r") as data_file:
        kernel = data_file["kernel"][:]
        sigma2 = data_file["sigma2"][()]
        observations = data_file["data"][local_slice]
    return kernel, sigma2, observations


def add_deconvolution_param(config_file_path: str, args: dict) -> None:
    config_file = open(config_file_path)
    params = json.load(config_file)

    args["split_coef"] = params["alpha"]
    args["reg_coef"] = params["regularizationCoefficient"]
    pass


def slice_obs_to_original(img_dims, kernel_dims):
    s = tuple(
        [
            np.s_[kernel_dims[d] // 2 : img_dims[d] + kernel_dims[d] // 2]
            for d in range(len(img_dims))
        ]
    )
    return s


def generate_gaussian_kernel(kernel_size, kernel_std) -> np.ndarray:
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
    w = scipy.signal.windows.gaussian(kernel_size, kernel_std)
    h = w[:, np.newaxis] * w[np.newaxis, :]
    h = h / np.sum(h)
    return h


def generate_gaussian_deconvolution_observations(
    original_path: str,
    kernel_dims: np.ndarray,
    kernel_std: float,
    snr: float,
    data_seed: int,
    obs_path: str,
    maximum: float = 1.0,
):
    #! backend/set_up must be done before the call
    from mcmc.operators.dft_convolution import (
        DftConvolution,
    )

    # FIXME: inconsistency in the type expected from img_dims and obs_dims, to be fixed (all tuple, or all ndarray)
    img_dims = load_img_size(original_path)
    kernel = generate_gaussian_kernel(kernel_dims, kernel_std)

    obs_dims = np.asarray(img_dims, dtype=int) + np.asarray(kernel_dims, dtype=int) - 1
    convolution_handler = DftConvolution(
        np.asarray(img_dims, dtype=int), kernel, (*obs_dims,)
    )

    pb_params = {}
    pb_params["kernel"] = kernel

    generate_observations(
        original_path,
        convolution_handler,
        snr,
        apply_gaussian_noise,
        data_seed,
        obs_path,
        maximum=1.0,
        problem_parameters=pb_params,
    )

    pass
