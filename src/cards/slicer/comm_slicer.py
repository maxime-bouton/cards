"""Generic slicer object to handle distributed arrays."""

from abc import ABC, abstractmethod

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)


class CommSlicer(ABC):
    """Extracting tile from or inserting into the global array for a given
    communicator based on several MPI processes.

    Attributes
    ----------
    grid_size : int
        Number of MPI processes underlying the communicator.
    global_buffer_size : np.array[int]
        Size of the global array slpit across processes.
    slice_global_buffer_to_tile : slice | list[slice]
        Slice to extract data tile from global array of size
        ``global_buffer_size``.
    """

    def __init__(
        self,
        grid_size,
        global_buffer_size,
    ):
        self.grid_size = grid_size
        self.global_buffer_size = global_buffer_size

        self.slice_global_buffer_to_tile = self._get_slice_global_buffer_to_tile()

    @abstractmethod
    def _get_slice_global_buffer_to_tile(self):
        """Create slice to insert tile into, or extract tile from, the full
        array."""
        pass
