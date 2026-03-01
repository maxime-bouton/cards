"""Useful metrics to assess reconstruction quality."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.

import numpy as np
from skimage.metrics import structural_similarity


def snr(x: np.ndarray, y: np.ndarray) -> float:
    r"""Compute the reconstruction Signal-to-Noise Ratio (SNR) with respect to a reference array ``x``.

    Parameters
    ----------
    x : np.ndarray
        Reference array.
    y : np.ndarray
        Estimated array.

    Returns
    -------
    float
        Reconstruction SNR.

    Raises
    ------
    ValueError
        Input arrays must have the same shape.
    """
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")

    noise = x - y
    signal_power = np.sum(x**2)
    noise_power = np.sum(noise**2)

    if noise_power == 0:
        return float("inf")

    return 10 * np.log10(signal_power / noise_power)


def ssim(x: np.ndarray, y: np.ndarray) -> float:
    r"""Compute the Structural Similarity Index (SSIM) between two images.

    Parameters
    ----------
    x : np.ndarray
        Reference image
    y : np.ndarray
        Estimated image.

    Returns
    -------
    float
        Structural Similarity Index betwwen the input images.
    """
    if x.shape != y.shape:
        raise ValueError("Input images must have the same shape.")

    return structural_similarity(
        x,
        y,
        data_range=x.max() - x.min(),
        channel_axis=-3 if len(x.shape) > 2 else None,
    )  # type: ignore
