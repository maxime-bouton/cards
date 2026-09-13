r"""Utility functions to generate synthetic data for the inpainting and
deconvolution experiments reported in :cite:p:`Bouton2026`."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from collections.abc import Callable, Sequence, Sized

import numpy as np

import cards.backend as xp
from cards.core.execution_context import ExecutionContext
from cards.operators.linear_operator import LinearOperator
from cards.utils.utils import expanded_left_view
from cards.utils.utils_img import normalize_ndarray


def generate_gaussian_kernel(
    kernel_width: int,
    kernel_std: float,
    dtype: type | None = None,
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
        Square Gaussian kernel, normalized so that :math:`\|h\|_1 = 1`.
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
    dtype: type | None = None,
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
    from cards.utils.blur_generator import MotionBlurKernel

    if xp.__name__ == "numpy":
        return MotionBlurKernel(
            (kernel_width, kernel_width), intensity, dtype, rng=rng
        ).kernel
    else:
        return xp.asarray(
            MotionBlurKernel(
                (kernel_width, kernel_width), intensity, dtype=dtype, rng=rng
            ).kernel
        )


def fit_kernel_shape(kernel: xp.ndarray, img_shape: Sized) -> xp.ndarray:
    r"""Broadcast the kernel to match the number of dimensions of the image.

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
    r"""Broadcast the mask to match the number of dimensions of the image.

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
    r"""Compute the slices to extract the original image from the linear convolution result.

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


def compute_sigma2_from_isnr(
    signal: xp.ndarray, isnr: float, ctx: ExecutionContext | None = None
) -> float:
    r"""Estimate the noise variance from the input signal and the desired input SNR.

    Parameters
    ----------
    signal: xp.ndarray
        The input signal (numpy/cupy array). If distributed, this is the local chunk.
    isnr: float
        The desired input SNR (in dB).
    ctx: ExecutionContext, optional
        Execution context object containing MPI properties (e.g., is_mpi, comm, rank).

    Returns
    -------
    float
        Corresponding noise variance (computed globally, returned identical on all ranks).
    """
    local_sq_norm = float(xp.linalg.norm(signal) ** 2)
    local_size = int(signal.size)

    if ctx is not None and ctx.is_mpi:
        from mpi4py import MPI

        global_sq_norm = ctx.comm.allreduce(local_sq_norm, op=MPI.SUM)
        global_size = ctx.comm.allreduce(local_size, op=MPI.SUM)
    else:
        global_sq_norm = local_sq_norm
        global_size = local_size

    if global_size == 0:
        raise ValueError("Global signal size is 0. Cannot compute variance.")

    return global_sq_norm / global_size / (10 ** (isnr / 10))


def generate_observations(
    img: xp.ndarray,
    operator: LinearOperator,
    apply_noise: Callable,
    rng: np.random.Generator,
    maximum: float = 1.0,
    **noise_args: float,
):
    r"""Generates and saves a deteriorated signal from the one given in entry.

    Parameters
    ----------
    img : xp.ndarray
        Input image.
    operator : LinearOperator
        Determinist deterioration operator.
    apply_noise : Callable
        Function to apply noise to the transformed image.
    rng : np.random.Generator
        Random number generator to generate synthetic data.
    maximum : float, optional
        Maximum value imposed for the ground truth image used to generate synthetic data.
    noise_args : dict, optional
        Dictionary containing noise specific parameters to be saved with the data.
    """
    normalized_img = normalize_ndarray(img, target_max=maximum)
    transformed_img = operator.forward(normalized_img)

    # retrieve potential noise parameters to be saved
    observations, *extra_params = apply_noise(transformed_img, rng, **noise_args)

    return observations, normalized_img, extra_params
