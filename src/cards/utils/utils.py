"""Utility functions to analyze generated data."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from collections.abc import Sequence
from typing import Any

import torch

import cards.backend as xp


def extract_subset_from_dict(d: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    """Extracts a subset of a dictionary based on the provided keys.

    Parameters
    ----------
    d : dict[str, Any]
        The original dictionary.
    keys : Sequence[str]
        The keys to extract.

    Returns
    -------
    dict
        A new dictionary containing only the specified keys and their values.
    """
    return {key: d[key] for key in keys}


def expand_shape_left(original_shape: Sequence[int], ndim: int) -> tuple[int, ...]:
    """Expand a shape to a specified number of dimensions on the left.

    Parameters
    ----------
    original_shape : Sequence[int]
        The original shape of the array.
    ndim : int
        The desired number of dimensions.

    Returns
    -------
    tuple[int,...]
        The expanded shape with the specified number of dimensions.
    """
    if len(original_shape) >= ndim:
        return tuple(original_shape)
    return (1,) * (ndim - len(original_shape)) + tuple(original_shape)


def expanded_left_view(array: xp.ndarray, ndim: int) -> xp.ndarray:
    """Expand the shape of an array to a specified number of dimensions on the left.

    Parameters
    ----------
    array : xp.ndarray
        The original array.
    ndim : int
        The desired number of dimensions.

    Returns
    -------
    xp.ndarray
        The expanded array with the specified number of dimensions.

    Notes
    -----
    This function uses broadcasting to expand the shape of the array without copying data.
    """

    if array.ndim >= ndim:
        return array
    return xp.broadcast_to(array, expand_shape_left(array.shape, ndim))


def xp2torch(x: xp.ndarray, add_batch: bool = True, torch_dtype=None):
    """Convert a ndarray to a PyTorch tensor.

    Parameters
    ----------
    x : xp.ndarray
        ndarray array to be converted.
    add_batch : bool, optional
        If True, add a batch dimension to the tensor (default is True).

    Returns
    -------
    torch.Tensor
        PyTorch tensor.
    """
    return torch.as_tensor(x[None, :] if add_batch else x).to(torch_dtype)


def torch2xp(x: torch.Tensor, remove_batch: bool = True, cp_dtype=None):
    """Convert a PyTorch tensor to a ndarray.

    Parameters
    ----------
    x : torch.Tensor
        PyTorch tensor to be converted.
    remove_batch : bool, optional
        If True, remove the batch dimension from the array (default is True).

    Returns
    -------
    xp.ndarray
        Converted ndarray.
    """
    return xp.asarray((x.squeeze(0) if remove_batch else x).detach(), dtype=cp_dtype)
