from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from mcmc.backend import xp


def load_img(
    path: str | Path,
    dtype: xp.dtype | None = None,
    key: str = "x",
) -> xp.ndarray:
    """Load an image from a file.

    Parameters
    ----------
    path : str or Path
        Path to the image file. Can be a .h5 file or an image file supported by PIL.
    dtype : xp.dtype, optional
        Data type of the loaded image. If None, the default dtype of the backend is used.
    key : str, optional
        Key to access the image data in the .h5 file, by default "x".

    Returns
    -------
    xp.ndarray
        The loaded image as a ndarray."""

    if Path(path).suffix == ".h5":
        with h5py.File(path) as file:
            img = xp.asarray(file[key])
    else:
        with Image.open(path, "r") as img_file:
            # NOTE: colored images are normalized and loaded in the shape (H, W, C)
            # better to load them in the format (C, H, W) directly ?
            img = xp.asarray(img_file, dtype=dtype) / 255.0

    return img


def read_img_shape(path: str | Path, key: str = "x") -> tuple[int, ...]:
    """Read the shape of an image from a file.

    Parameters
    ----------
    path : str or Path
        Path to the image file. Can be a .h5 file or an image file supported by PIL.
    key : str, optional
        Key to access the image data in the .h5 file, by default "x".

    Returns
    -------
    tuple[int, ...]
        The shape of the image as a tuple of integers.
    """
    if Path(path).suffix == ".h5":
        with h5py.File(path) as file:
            return file[key].shape  # type: ignore
    else:
        with Image.open(path) as img:
            # NOTE: lazy loading of the size (image is not loaded in memory)
            w, h = img.size
            channels = len(img.getbands())
            return (h, w, channels)


def read_dtype(path: str | Path, key: str = "x") -> np.dtype:
    """Read the dtype of a variable from a `.h5` file.

    Parameters
    ----------
    path : str or Path
        Path to the `.h5` file.
    key : str, optional
        Key to access the image data in the .h5 file, by default "x".

    Returns
    -------
    np.dtype
        The python-compatible numpy dtype of the image.
    """
    if Path(path).suffix == ".h5":
        with h5py.File(path) as file:
            return file[key].dtype  # type: ignore
    else:
        raise ValueError(
            "The provided path does not point to a .h5 file. "
            "This function is only applicable for .h5 files."
        )


def normalize_ndarray(
    x: xp.ndarray,
    maximum: float = 1.0,
    minimum: float = 0.0,
) -> xp.ndarray:
    """Normalize an ndarray to a specified range.

    Parameters
    ----------
    x : xp.ndarray
        The input ndarray to be normalized.
    maximum : float, optional
        The maximum value of the normalized ndarray, by default 1.0
    minimum : float, optional
        The minimum value of the normalized ndarray, by default 0.0

    Returns
    -------
    xp.ndarray
        The normalized ndarray, scaled to the range [minimum, maximum].
    """
    a = minimum
    b = maximum
    return a + (b - a) * (x - xp.min(x)) / (xp.max(x) - xp.min(x))
