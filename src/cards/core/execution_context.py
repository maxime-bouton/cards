"""Defines the execution context for the CARDS framework"""

from typing import Literal

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
