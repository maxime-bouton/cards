from pathlib import Path

import h5py
import numpy as np

from cards.operators.dft_convolution import DftConvolution
from cards.utils.path_builder import (
    deconvolution_str,
    gaussian_str,
    obs_dir,
)
from cards.utils.utils import extract_subset_from_dict
from cards.utils.utils_img import read_dtype, read_img_shape
from cards.utils.utils_observations import (
    apply_target_gaussian_noise,
    fit_kernel_shape,
    generate_and_save_observations,
    generate_gaussian_kernel,
    generate_motion_kernel,
    slice_linear_conv_to_original,
)

from cards.backend import xp


def load_from_h5(
    filename: str | Path,
) -> tuple[xp.ndarray, float, tuple[int, ...], tuple[int, ...]]:
    """Load the kernel, sigma2, and observation shape from an HDF5 file.

    Parameters
    ----------
    filename : str or Path
        Path to the HDF5 file containing the kernel, sigma2, and observations.

    Returns
    -------
    tuple[xp.ndarray, float, tuple[int, ...], tuple[int, ...]]
        A tuple containing the kernel as a cupy array, the sigma2 value as a float,
        the shape of the ground truth as a tuple of integers,
        and the shape of the observations as a tuple of integers.
    """
    with h5py.File(filename, "r") as data_file:
        kernel = xp.asarray(data_file["kernel"])
        sigma2 = data_file["sigma2"][()]  # type: ignore
        gt_shape = data_file["x"].shape  # type: ignore
        obs_shape = data_file["y"].shape  # type: ignore
    return kernel, sigma2, gt_shape, obs_shape  # type: ignore


def gaussian_deconvolution_params(params: dict) -> dict:
    if "denoiser_params" in params:
        return extract_subset_from_dict(params, ["reg_coef", "denoiser_params"])
    else:
        return extract_subset_from_dict(params, ["reg_coef", "split_coef"])


def define_slices(params: dict) -> dict:
    img_size = read_img_shape(params["original_img_path"])
    # TODO: find a more elegant way to deal with image of different dimensions
    kernel_size = np.asarray([1] * (len(img_size) - 2) + [params["kernel"]["size"]] * 2)
    return {"slices": slice_linear_conv_to_original(img_size, kernel_size)}


def generate_gaussian_deconvolution_observations(
    original_img_path: str,
    kernel_params: dict,
    isnr: float,
    data_seed: int,
    obs_path: str,
    maximum: float = 1.0,
):
    gt_size = np.asarray(read_img_shape(original_img_path))
    dtype = read_dtype(original_img_path)

    rng = np.random.default_rng(data_seed)

    if kernel_params["type"] == "motion":
        kernel = generate_motion_kernel(
            kernel_params["size"],
            kernel_params["intensity"],
            dtype,
            rng,
        )
    else:
        kernel = generate_gaussian_kernel(
            kernel_params["size"],
            kernel_params["std"],
            dtype,
        )

    params_saved = {"kernel": kernel}

    reshaped_kernel = fit_kernel_shape(kernel, gt_size)

    obs_dims = gt_size.copy()
    # convolution affects only the last two dimensions (i.e., spatial dimensions)
    obs_dims[-2:] += np.asarray(kernel.shape, dtype=int) - 1

    convolution_handler = DftConvolution(gt_size, reshaped_kernel, obs_dims)

    generate_and_save_observations(
        original_img_path,
        obs_path,
        convolution_handler,
        apply_target_gaussian_noise,
        data_seed,
        params_saved,
        maximum,
        isnr=isnr,
    )


def application_params_dir(params: dict) -> str:
    if "denoiser_params" in params:
        return "."
    return f"split{params['split_coef']}".replace(".", "_")


def build_obs_and_model_paths(params: dict) -> tuple[str, str]:
    noise_str = gaussian_str(params)
    application_str = deconvolution_str(params)
    obs_f = f"gaussian-deconvolution/{obs_dir(params, application_str, noise_str)}"
    model_params_f = application_params_dir(params)

    return obs_f, model_params_f
