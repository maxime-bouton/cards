from pathlib import Path
from typing import Callable, Sequence, Sized

import h5py
import numpy as np

from mcmc.backend import xp
from mcmc.operators.linear_operator import LinearOperator
from mcmc.utils.utils import expanded_left_view
from mcmc.utils.utils_img import load_img, normalize_ndarray

# TODO: `rng` should be compatible with both numpy and cupy
# create rng abstraction layer that can handle both numpy and cupy/torch generators ?


def generate_gaussian_kernel(
    kernel_width: int,
    kernel_std: float,
    dtype: xp.dtype | None = None,
) -> xp.ndarray:
    r"""Generate a square normalized 2D Gaussian kernel.

    Parameters
    ----------
    kernel_width : int
        Size of one dimension of the kernel.
    kernel_std : float
        Standard deviation of the Gaussian kernel.
    dtype : type, optional
        Data type of the kernel, by default `xp.float64`.

    Note
    ----
    Equivalent to the ``fspecial('gaussian', ...)`` function in Matlab.

    Returns
    -------
    xp.ndarray
        Square Gaussian kernel with :math:`\|h\|_1 = 1`.
    """
    if xp.__name__ == "numpy":
        from scipy.signal.windows import gaussian
    else:
        from cupyx.scipy.signal.windows import gaussian

    w = gaussian(kernel_width, kernel_std).astype(dtype)
    h = w[:, None] * w[None, :]
    return h / xp.sum(h)


def generate_motion_kernel(
    kernel_width: int,
    intensity: float,
    dtype: xp.dtype | None = None,
    rng: np.random.Generator | None = None,
) -> xp.ndarray:
    r"""Generate a square normalized 2D motion kernel.

    Parameters
    ----------
    kernel_width : int
        Size of one dimension of the kernel.
    intensity : float
        Intensity of the motion blur.
    dtype : type, optional
        Data type of the kernel, by default `xp.float64`.
    rng : np.random.Generator, optional
        Random number generator for reproducibility, by default None.

    Returns
    -------
    xp.ndarray
        Square motion kernel with :math:`\|h\|_1 = 1`.
    """
    from mcmc.utils.blur_generator import MotionBlurKernel

    if xp.__name__ == "numpy":
        return MotionBlurKernel((kernel_width, kernel_width), intensity, rng).kernel
    else:
        return xp.asarray(
            MotionBlurKernel(
                (kernel_width, kernel_width), intensity, dtype=dtype, rng=rng
            ).kernel
        )


def fit_kernel_shape(kernel: xp.ndarray, img_shape: Sized) -> xp.ndarray:
    """Broadcast the kernel to match the number of dimensions of the image.

    Parameters
    ----------
    kernel : xp.ndarray
        The kernel to be broadcasted.
    img_shape : Sized
        The shape of the image to which the kernel should be broadcasted.

    Returns
    -------
    xp.ndarray
        The broadcasted kernel.
    """
    if len(img_shape) < 2:
        raise ValueError(
            "Image shape must have at least two dimensions (height and width)."
        )
    return expanded_left_view(kernel, len(img_shape))


def fit_mask_shape(
    mask: xp.ndarray,
    img_shape: Sized,
) -> xp.ndarray:
    """Broadcast the mask to match the number of dimensions of the image.

    Parameters
    ----------
    mask : xp.ndarray
        The mask to be broadcasted.
    img_shape : Sized
        The shape of the image to which the mask should be broadcasted.

    Returns
    -------
    xp.ndarray
        The broadcasted mask.
    """
    if len(img_shape) < 2:
        raise ValueError(
            "Image shape must have at least two dimensions (height and width)."
        )
    return expanded_left_view(mask, len(img_shape))


def slice_linear_conv_to_original(
    img_shape: Sequence[int],
    kernel_shape: Sequence[int],
) -> tuple[slice, ...]:
    """Compute the slices to extract the original image from the linear convolution result.

    Parameters
    ----------
    img_shape : Sequence[int]
        Shape of the original image.
    kernel_shape : Sequence[int]
        Shape of the kernel used for convolution.

    Returns
    -------
    tuple[slice, ...]
        Slices to extract the original image from the linear convolution result.
    """
    return tuple(np.s_[k // 2 : i + k // 2] for k, i in zip(kernel_shape, img_shape))


def estimate_sigma2_from_isnr(signal: xp.ndarray, isnr: float) -> float:
    """Estimate the noise variance from the input signal and the desired iSNR.

    Parameters
    ----------
    signal: xp.ndarray
        The input signal (numpy array).
    isnr: float
        The desired iSNR (in dB).

    Returns:
    -------
    float
        The estimated noise variance.
    """
    return float(xp.linalg.norm(signal) ** 2 / signal.size / (10 ** (isnr / 10)))


def apply_poisson_noise(
    signal: xp.ndarray,
    rng: np.random.Generator,
    dynamic_range: float = 1.0,
) -> tuple[xp.ndarray, dict[str, float]]:
    """Apply Poisson noise to the input signal based on the specified scale.

    Parameters
    ----------
    signal: xp.ndarray
        The input signal.
    rng: np.random.Generator
        Random number generator for reproducibility.
    scale: float, optional
        The scale parameter for the Poisson distribution, by default 1.0.

    Returns:
    -------
    tuple[xp.ndarray, dict[str, float]]

    """

    if not isinstance(signal, np.ndarray):
        signal = signal.get()

    return xp.asarray(
        rng.poisson(np.maximum(signal, 0) * dynamic_range),
        dtype=signal.dtype,
    ), {"dynamic_range": dynamic_range}


def apply_gaussian_noise(
    signal: xp.ndarray,
    rng: np.random.Generator,
    sigma2: float,
) -> xp.ndarray:
    """Apply Gaussian noise to the input signal based on the specified variance.

    Parameters
    ----------
    signal: xp.ndarray
        The input signal.
    rng: np.random.Generator
        Random number generator for reproducibility.
    sigma2: float
        The variance of the Gaussian noise to be applied.

    Returns:
    -------
    xp.ndarray
        The noisy signal.
    """
    return (
        signal
        + xp.asarray(rng.standard_normal(signal.shape, signal.dtype)) * sigma2**0.5
    )


def apply_target_gaussian_noise(
    signal: xp.ndarray,
    rng: np.random.Generator,
    isnr: float,
) -> tuple[xp.ndarray, dict[str, float]]:
    """Apply noise to the input signal based on the desired iSNR.

    Parameters
    ----------
    signal: xp.ndarray
        The input signal.
    rng: np.random.Generator
        Random number generator for reproducibility.
    isnr: float
        The desired iSNR (in dB).

    Returns:
    -------
    tuple[xp.ndarray, dict[str, float]]
        The noisy signal and a dictionary containing the estimated noise variance.
    """
    sigma2 = estimate_sigma2_from_isnr(signal, isnr)
    return apply_gaussian_noise(signal, rng, sigma2), {"sigma2": sigma2}


def generate_and_save_observations(
    original_img_path: str | Path,
    observations_path: str | Path,
    operator: LinearOperator,
    apply_noise: Callable,
    seed_data: int,
    params_saved: dict,
    maximum: float = 1.0,
    **noise_args: float,
):
    """Generates and saves a deteriorated signal from the one given in entry.

    Parameters
    ----------
    original_img_path : str
        Path to the file containing the ground truth.
    observations_path : str
        Path to the file where to save the generated data.
    operator : LinearOperator
        Determinist deterioration operator.
    apply_noise : Callable
        Function to apply noise to the transformed image.
    seed_data : int
        Seed.
    maximum : float, optional
        Maximum value imposed for the ground truth image used to generate synthetic data.
    noise_args : dict, optional
        Dictionnary containing noise specific parameters to be saved with the data.
    """

    img = load_img(original_img_path)
    normalized_img = normalize_ndarray(img, maximum=maximum)

    rng = np.random.default_rng(seed_data)

    transformed_img = operator.forward(normalized_img)

    # retrieve potential noise parameters to be saved
    observations, *extra_params = apply_noise(transformed_img, rng, **noise_args)

    params_saved.update(**noise_args)

    # NOTE: the tuple `extra_params` is either empty or containing a single dictionary
    # TODO: abstract the noise application to avoid this pattern
    if extra_params and isinstance(extra_params[0], dict):
        params_saved.update(**extra_params[0])

    with h5py.File(observations_path, "w") as file:
        file["x"] = (
            normalized_img
            if isinstance(normalized_img, np.ndarray) or np.isscalar(normalized_img)
            else normalized_img.get()
        )
        file["y"] = (
            observations
            if isinstance(observations, np.ndarray) or np.isscalar(observations)
            else observations.get()
        )
        file["seed_data"] = seed_data

        for key, value in params_saved.items():
            file[key] = (
                value
                if isinstance(value, np.ndarray) or np.isscalar(value)
                else value.get()
            )
