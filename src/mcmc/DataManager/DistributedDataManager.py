"""
Object than handles any reading/writing on disk with parallel memory acces.
"""

import h5py
import numpy as np

from mcmc.DataManager.warmstart_rng_mpi import load_rng_np_mpi, save_rng_np_mpi

# ! need a base class setting up the interface, and create subclasses through inheritance (numpy-based, torch-based)


class DistributedDataManager:
    def save_dict(
        self, data: dict, file: h5py.File, global_sizes: dict, slices: dict
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
            #! acces mode must be set in parallel so that variables can be created in parallel
            dset = file.create_dataset(key, global_sizes[key], dtype=data[key].dtype)
            dset[slices[key]] = data[key]

    def save_array(
        self,
        data: np.ndarray,
        global_size: np.ndarray,
        slices: slice,
        file: h5py.File,
        name: str,
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
        dset = file.create_dataset(name, global_size, dtype=data.dtype)
        dset[slices] = data

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

    def load_h5(self, file: h5py.File, slices: dict) -> dict:
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
        for key in slices.keys():
            data[key] = file[key][slices[key]][:]
        return data

    def save_rng(
        self, rng: np.random.Generator, file: h5py.File, rank: int, comm_size: int
    ) -> None:
        """save_rng Seva the internal state of all the generator used along all the processes.

        Parameters
        ----------
        comm : MPI.Comm
            Current MPI communicator.
        rng : np.random.Generator
            Local random number generator.
        file : h5py.File
            File to be written on.
        """
        save_rng_np_mpi(rank, comm_size, rng, file)
        return

    def load_rng(self, rng: np.random.Generator, file: h5py.File, rank: int) -> None:
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
        load_rng_np_mpi(rank, rng, file)
        return
