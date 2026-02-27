import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def snr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Signal-to-Noise Ratio (SNR) between two images."""
    if x.shape != y.shape:
        raise ValueError("Input images must have the same shape.")

    noise = x - y
    signal_power = np.sum(x**2)
    noise_power = np.sum(noise**2)

    if noise_power == 0:
        return float("inf")  # Infinite SNR if there's no noise

    return 10 * np.log10(signal_power / noise_power)


def ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Structural Similarity Index (SSIM) between two images."""
    if x.shape != y.shape:
        raise ValueError("Input images must have the same shape.")

    return structural_similarity(
        x,
        y,
        data_range=1.0,
        channel_axis=-3 if len(x.shape) > 2 else None,
    )  # type: ignore


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    if x.shape != y.shape:
        raise ValueError("Input images must have the same shape.")

    return peak_signal_noise_ratio(
        x,
        y,
        data_range=1.0,
    )
