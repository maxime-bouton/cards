r"""Utility functions to load and normalize images."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.

from pathlib import Path

import h5py
from PIL import Image

import cards.backend as xp


def load_img(
    path: str | Path,
    dtype: xp.dtype | None = None,
    key: str = "x",
) -> xp.ndarray:
    r"""Load an image from a file.

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
    r"""Read the shape of an image from a file.

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
            return file[key].shape
    else:
        with Image.open(path) as img:
            # NOTE: lazy loading of the size (image is not loaded in memory)
            w, h = img.size
            channels = len(img.getbands())
            return (h, w, channels)


def read_dtype(path: str | Path, key: str = "x") -> type:
    r"""Read the dtype of a variable from a `.h5` file.

    Parameters
    ----------
    path : str or Path
        Path to the `.h5` file.
    key : str, optional
        Key to access the image data in the .h5 file, by default "x".

    Returns
    -------
    type
        The python-compatible numpy type of the image.
    """
    if Path(path).suffix == ".h5":
        with h5py.File(path) as file:
            return file[key].dtype
    else:
        raise ValueError(
            "The provided path does not point to a .h5 file. "
            "This function is only applicable for .h5 files."
        )


def normalize_ndarray(
    x: xp.ndarray,
    target_max: float = 1.0,
    target_min: float = 0.0,
) -> xp.ndarray:
    r"""Normalize an input array to a specified range.

    Parameters
    ----------
    x : xp.ndarray
        The input ndarray to be normalized.
    target_max : float, optional
        Target maximum value after normalization, by default 1.0
    target_main : float, optional
        Target minimum value after normalization, by default 0.0

    Returns
    -------
    xp.ndarray
        The normalized ndarray, scaled to the range [minimum, maximum].
    """
    return target_min + (target_max - target_min) * (x - xp.min(x)) / (
        xp.max(x) - xp.min(x)
    )
