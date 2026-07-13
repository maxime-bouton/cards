r"""Generic slicer class to handle distributed tensors."""

from abc import ABC, abstractmethod

import numpy as np


class BaseCommSlicer(ABC):
    r"""Extracting tile from or inserting into the global array for a given
    communicator based on several MPI processes.

    Parameters
    ----------
    grid_size : numpy.ndarray[int]
        Number of processes along each axis of the Cartesian grid, as returned by ``np.array(MPI.Compute_dims(size, ndims), dtype="i")``.
    global_buffer_size : np.array[int]
        Shape of a `d`-dimensional array (tensor) decomposed over the Cartesian grid of workers considered.

    Attributes
    ----------
    grid_size : numpy.ndarray[int]
        Number of processes along each axis of the Cartesian grid, as returned by ``np.array(MPI.Compute_dims(size, ndims), dtype="i")``.
    global_buffer_size : np.array[int]
        Shape of a `d`-dimensional array (tensor) decomposed over the Cartesian grid of workers considered.
    slice_global_buffer_to_tile : slice | list[slice]
        Slice to extract data tile from a tensor of size ``global_buffer_size`` tessellated across the processes.
    """

    def __init__(
        self,
        grid_size: np.ndarray,
        global_buffer_size: np.ndarray,
    ) -> None:
        self.grid_size = grid_size
        self.global_buffer_size = global_buffer_size

        self.slice_global_buffer_to_tile = self._get_slice_global_buffer_to_tile()

    @abstractmethod
    def _get_slice_global_buffer_to_tile(self) -> slice:
        r"""Returns a slice to insert or extract a tile from the tensor tessellated across the workers."""
