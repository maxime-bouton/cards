"""Defines the execution context for the CARDS framework"""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from typing import Literal

import numpy as np
import torch

import cards.backend as xp


class ExecutionContext:
    r"""Class representing the execution context for the CARDS framework.

    It encapsulates the execution mode (serial or MPI) and device type (CPU or GPU). It
    triggers the proper backend initialization for the specified device type and stores
    relevant information such as the MPI communicator, rank, and total number of workers.

    Parameters
    ----------
    mode : {"serial", "mpi"}
        Execution mode.
    device : {"cpu", "gpu"}
        Device type.

    Attributes
    ----------
    mode : str
        Execution mode, either "serial" or "mpi".
    device : str
        Device type, either "cpu" or "gpu".
    is_mpi : bool
        Whether the execution mode is "mpi".
    is_gpu : bool
        Whether the device type is "gpu".
    comm : MPI.Comm | None
        MPI communicator, if "mpi" mode, otherwise None.
    rank : int
        MPI rank of the current worker. 0 if "serial".
    comm_size : int
        Total number of MPI workers. 1 if "serial".
    is_master : bool
        Whether the current worker is the master (rank 0). True if "serial".

    Methods
    -------
    generate_grid_size()
        Generate the MPI grid size for a given data dimension and splitting strategy.
    """

    def __init__(
        self,
        mode: Literal["serial", "mpi"],
        device: Literal["cpu", "gpu"],
    ) -> None:

        if mode not in ("serial", "mpi"):
            raise ValueError(f"Invalid mode: '{mode}'. Expected 'serial' or 'mpi'.")
        if device not in ("cpu", "gpu"):
            raise ValueError(f"Invalid device: '{device}'. Expected 'cpu' or 'gpu'.")

        self._mode = mode
        self._device = device

        self._comm = None
        self._rank = 0
        self._comm_size = 1

        self._setup_environment()

    def __str__(self) -> str:
        if self.is_mpi:
            return f"{self._mode}-{self._device}_{self._comm_size}"
        return f"{self._mode}-{self._device}"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_mpi(self) -> bool:
        return self._mode == "mpi"

    @property
    def is_gpu(self) -> bool:
        return self._device == "gpu"

    @property
    def comm(self):
        return self._comm

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def comm_size(self) -> int:
        return self._comm_size

    @property
    def is_master(self) -> bool:
        return self._rank == 0

    def _setup_environment(self) -> None:
        """Initialize MPI and Torch/Cupy backends."""
        if self.is_mpi:
            from mpi4py import MPI

            self._comm = MPI.COMM_WORLD
            self._rank = self._comm.Get_rank()
            self._comm_size = self._comm.Get_size()

        if self.is_gpu:
            xp.set_backend("cupy")
            gpu_id = self._rank % xp.cuda.runtime.getDeviceCount()
            xp.cuda.Device(gpu_id).use()
            torch.cuda.set_device(gpu_id)
            torch.set_default_device("cuda")

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        torch.use_deterministic_algorithms(True)

    def generate_grid_size(
        self,
        dims: int,
        strategy: Literal["best", "auto"] = "best",
        spatial_dims: int = 2,
    ) -> np.ndarray:
        """Compute the size of the MPI grid partitioning the data of given dimensions.

        Parameters
        ----------
        dims : int
            Total number of dimensions of the data to be partitioned.
        strategy : Literal["best", "auto"], optional
            Strategy to use for computing the grid size over the spatial axes.
            If `best`, the partitioning is only done along the first spatial axis.
            If `auto`, the grid is computed using MPI's :meth:`Compute_dims` function
            distributed across all spatial dimensions.
        spatial_dims : int, optional
            Number of spatial dimensions at the end of the shape to be partitioned.
            Default is 2 (e.g., Height, Width).

        Returns
        -------
        np.ndarray
           The size of the grid partitioning the data.
        """
        if not self.is_mpi:
            return np.ones(dims, dtype=int)

        if spatial_dims < 1:
            raise ValueError(
                "There must be at least one spatial dimension to partition."
            )

        if dims < spatial_dims:
            raise ValueError(
                f"Data dims ({dims}) cannot be less than spatial dims ({spatial_dims})."
            )

        # first dimensions are not partitioned (no mpi communication along them)
        prefix = [1] * (dims - spatial_dims)
        if strategy == "best":
            # partition only along the first spatial dimension, keep the rest as 1
            suffix = [self.comm_size] + [1] * (spatial_dims - 1)
        else:
            from mpi4py import MPI

            # MPI distributes the ranks across ALL spatial dimensions
            suffix = MPI.Compute_dims(self.comm_size, spatial_dims)
        return np.asarray(prefix + suffix, dtype=int)
