"""
Object than handles any reading/writing on disk with parallel memory acces.
"""

import sys
from typing import Optional

import h5py
import numpy as np
import torch

from mcmc.data_manager.warmstart_rng import (
    load_rng_np,
    load_rng_offset_torch,
    save_rng_np,
    save_rng_offset_torch,
)
from mcmc.data_manager.warmstart_rng_mpi import load_rng_np_mpi, save_rng_np_mpi

# ! need a base class setting up the interface, and create subclasses through inheritance (numpy-based, torch-based)


class DataManager:
    def save_dict(
        self,
        data: dict,
        file: h5py.File,
        global_sizes: Optional[dict] = None,
        slices: Optional[dict] = None,
    ) -> None:
        """save_dict Saves the dictionnary given in entry in the .h5 file given in entry.
        It expects the given file to be open in paralell mode.

        Parameters
        ----------
        data : dict
            Dictionnary containing data to write on disk.
        file : h5py.File
            File on wich the data will be written.
        global_size:
            Dictionnary containing the golbal dimensions of the buffers.
        slices: dict
            Dictionnary containing the indexes of the vertices delimiting the position of the local buffer in the global buffer.
        """
        for key in data:
            if global_sizes is None:
                buffer_size = data[key].shape
            else:
                buffer_size = global_sizes[key]

            if slices is None:
                local_slice = slice(None)
            else:
                local_slice = slices[key]
            dset = file.create_dataset(
                name=key, shape=buffer_size, dtype=data[key].dtype
            )
            dset[local_slice] = data[key]

    def save_array(
        self,
        data: np.ndarray,
        file: h5py.File,
        name: str,
        global_size: Optional[np.ndarray] = None,
        local_slice: Optional[slice] = slice(None),
    ) -> None:
        """save_array Save the array given in entry in the .h5 file given in entry.

        Parameters
        ----------
        data : np.ndarray
            Array of data to write on file.
        global_size:
            Golbal dimensions of the buffers.
        slices: slice
            Indexes of the vertices delimiting the position of the local buffer in the global buffer.
        file : h5py.File
            File on wich we write the data.
        name : str
            Name of the datafield in the file.
        """
        if global_size is None:
            global_size = data.shape
        dset = file.create_dataset(name, global_size, dtype=data.dtype)
        dset[local_slice] = data

    def save_seed(self, seed: int, rank: int, comm_size: int, file: h5py.File) -> None:
        """save_seed Save the seeds used on each process. It expects the given file to be open in paralell mode.

        Parameters
        ----------
        seed : int
            Local seed.
        rank : int
            Rank of the process.
        comm_size : int
            Number of process.
        file : h5py.File
            File to be written on.
        """
        dset = file.create_dataset("seed", (comm_size,), dtype=int)
        dset[rank] = seed

    def save_local_array(self, data: np.ndarray, name: str, file: h5py.File) -> None:
        """save_local_array Save an array on a .h5 file. It expects the given file to be open in serial mode.

        Parameters
        ----------
        data : np.ndarray
            Local array.
        name : str
            Name of the variable.
        file : h5py.File
            File to be written on.
        """
        dset = file.create_dataset(name, data.shape, dtype=data.dtype)
        dset[:] = data
        return

    def save_thread_array(
        self, data: np.ndarray, rank: int, comm_size: int, name: str, file: h5py.File
    ) -> None:
        """save_thread_array Simultaneously save an array along each thread.

        Parameters
        ----------
        data : np.ndarray
            Local array.
        rank : int
            Rank of the current thread.
        comm_size : int
            Number of thread available int he commuicator.
        name : str
            Name of the datafield.
        file : h5py.File
            File where to writte the data.
        """
        dset = file.create_dataset(name, (comm_size, *data.shape), dtype=data.dtype)
        dset[rank, ...] = data
        return

    def save_thread_scalar(
        self, data: float, rank: int, comm_size: int, name: str, file: h5py.File
    ) -> None:
        """save_thread_scalar Simultaneously save a scalar along each thread.

        Parameters
        ----------
        data : np.ndarray
            Local scalar.
        rank : int
            Rank of the current thread.
        comm_size : int
            Number of thread available int he commuicator.
        name : str
            Name of the datafield.
        file : h5py.File
            File where to writte the data.
        """
        dset = file.create_dataset(name, comm_size, dtype=data.dtype)
        dset[rank] = data
        return

    def load_h5(self, file: h5py.File, slices: Optional[dict] = None) -> dict:
        """load_h5 Read a .5 file and load the local value of the array on each process.
        It expects the given file to be open in paralell mode.

        Parameters
        ----------
        file : h5py.File
            File to be read.
        slices : dict
            Dictionnary containing the slices of each variables, different on each thread.

        Returns
        -------
        dict
            Dictonnary containing the local value of each variable.
        """
        data = {}
        if slices is None:
            for key in file.keys():
                if file[key].size > 1:
                    data[key] = file[key][:]
                else:
                    data[key] = file[key][()]
        else:
            for key in slices.keys():
                data[key] = file[key][slices[key]][:]
        return data

    def save_rng(
        self,
        rng: np.random.Generator,
        file: h5py.File,
        rank: int = 0,
        comm_size: int = 0,
    ) -> None:
        """save_rng Save the internal state of all the generator used along all the processes.

        Parameters
        ----------
        comm : MPI.Comm
            Current MPI communicator.
        rng : np.random.Generator
            Local random number generator.
        file : h5py.File
            File to be written on.
        """
        if comm_size == 0:
            save_rng_np(rng, file)
        else:
            save_rng_np_mpi(rank, comm_size, rng, file)
        return

    def load_rng(
        self, rng: np.random.Generator, file: h5py.File, rank: Optional[int] = None
    ) -> None:
        """load_rng Load the internal state of the random number generator for each process.

        Parameters
        ----------
        rng : np.random.Generator
            Local random number generator.
        file : h5py.File
            File to be read.
        rank : int
            Rank of the process.
        """
        if rank is None:
            load_rng_np(rng, file)
        else:
            load_rng_np_mpi(rank, rng, file)
        return

    def save_rng_torch(
        self,
        rng: torch._C.Generator,
        seed: int,
        h5file: h5py.File,
        gpu_id: Optional[int] = None,
        nb_gpu: int = 0,
    ) -> None:
        r"""Save current state of a pytorch random number generator in a .h5 file
        using the offset from the initial seed state.

        Parameters
        ----------
        rng : torch._C.Generator
            Pytorch random number generator on the GPU.
        seed : int
            Seed used to initialize the generator.
        h5file : h5py.File
            Handle to a `.h5` file to save the state of the generator.
        gpu_id : int
            Identifaint of the current device.
        comm_size : int
            Number of GPU available.
        Note
        ----
        Requires ``pythorch>=2.5``. Only supported for generators on the GPU.
        """

        if gpu_id is None:
            save_rng_offset_torch(rng, seed, h5file)
        else:
            seed_array = int_to_array(seed)
            dset_seed = h5file.create_dataset(
                "seed", (*seed_array.shape, nb_gpu), dtype=seed_array.dtype
            )
            dset_seed[:, gpu_id] = seed_array

            offset_array = int_to_array(rng.get_offset())
            dset_offset = h5file.create_dataset(
                "offset",
                (*offset_array.shape, nb_gpu),
                dtype=offset_array.dtype,
            )
            dset_offset[:, gpu_id] = offset_array
        return

    def load_rng_torch(
        self, rng: torch.Generator, h5file: h5py.File, gpu_id: Optional[int] = None
    ):
        r"""Load the state of several pytorch random number generators from a .h5 file using the offset from an initial seed state.

        Parameters
        ----------
        rng : torch._C.Generator
            Pytorch local random number generator.
        h5file : h5py.File
            Handle to a `.h5` file to save the state of the generator.
        gpu_id : int
            Identifiant of the current device.

        Note
        ----
        Requires ``pythorch>=2.5``. Only supported for generators on the GPU.
        """
        # ! an offset is relative to some initial seed, which also needs to be
        # ! loaded and set
        if gpu_id is None:
            load_rng_offset_torch(rng, h5file)
        else:
            seed = array_to_int(h5file["seed"][:, gpu_id])
            rng.manual_seed(seed)
            offset = array_to_int(h5file["offset"][:, gpu_id])
            rng.set_offset(offset)
        return


def int_to_array(n: int) -> np.ndarray[np.uint8]:
    r"""Convert a built-on Python ``int`` to an array which can be saved in
    hdf5.

    Parameters
    ----------
    n : int
        Input `int`` value.

    Returns
    -------
    n_array
        Output array conversion.

    Note
    ----
    Built-in Python ``int``s, as those used to describe the state of a numpy
    random number generator, can be very large. They need to be converted to
    hexadecimal format, and from that into an array of ``np.uint8``, in order
    to be saved into an .h5 file.
    """
    # Reference:
    # https://docs.python.org/3/library/stdtypes.html#int.to_bytes
    # ! need 32 bytes in length: otherwise, the inverse operation
    # ! int.from_bytes(state_array,sys.byteorder) does not coincide with the
    # ! original int value
    # ! entries in the resulting array are of type np.uint8
    int_array = np.array(bytearray(n.to_bytes(32, sys.byteorder)))
    return int_array


def array_to_int(n_array: np.ndarray[np.uint8]) -> int:
    r"""Convert a numpy array of ``np.uint8`` back to a built-in Python ``int``.

    Inverse operation of :func:`mcmc.DataManager.warmstart_rng.int_to_array`.

    Parameters
    ----------
    n_array : np.ndarray[np.uint8]
        Input array.

    Returns
    -------
    int
        Output `int`` value.
    """
    return int.from_bytes(n_array, sys.byteorder)
