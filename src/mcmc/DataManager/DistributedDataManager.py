"""
Object than handles any reading/writing on disk with parallel memory acces.
"""

import h5py
import numpy as np
import sys


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
            #! acces mode must be set in parallel
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
        slices: dict
            Indexes of the vertices delimiting the position of the local buffer in the global buffer.
        file : h5py.File
            File on wich we write the data.
        name : str
            Name of the datafield in the file.
        """
        dset = file.create_dataset(name, global_size, dtype=data.dtype)
        dset[slices] = data

    def save_seed(self, seed: int, rank: int, comm_size: int, file: h5py.File):
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
        dset = file.create_dataset("seed", np.asarray([comm_size]), dtype=int)
        dset[rank] = seed

    def save_local_array(self, data: np.ndarray, name: str, file: h5py.File):
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
        file[name] = data

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
            data[key] = np.asarray(file[key][slices[key]], dtype=file[key].dtype)
        return data

    def save_rng(
        self, rng: np.random.Generator, file: h5py.File, rank: int, comm_size: int
    ) -> None:
        """save_rng Seva the internal state of all the generator used along all the processes.

        Parameters
        ----------
        rng : np.random.Generator
            Local random number generator.
        file : h5py.File
            File to be written on.
        rank : int
            Rank of the process.
        comm_size : int
            Number of processes.
        """

        state_array = np.array(
            bytearray(
                rng.bit_generator.__getstate__()[0]["state"]["state"].to_bytes(
                    32, sys.byteorder
                )
            )
        )
        inc_array = np.array(
            bytearray(
                rng.bit_generator.__getstate__()[0]["state"]["inc"].to_bytes(
                    32, sys.byteorder
                )
            )
        )

        size = np.asarray([comm_size, *state_array.shape])
        dset = file.create_dataset("rng_state_array", size, dtype=np.uint8)
        dset[rank] = state_array

        size = np.asarray([comm_size, *inc_array.shape])
        dset = file.create_dataset("rng_inc_array", size, dtype=np.uint8)
        dset[rank] = inc_array

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

        new_state = rng.bit_generator.__getstate__()

        loaded_state = file["rng_state_array"][rank]
        loaded_inc = file["rng_inc_array"][rank]

        new_state_state = int.from_bytes(loaded_state, sys.byteorder)
        new_state_inc = int.from_bytes(loaded_inc, sys.byteorder)
        new_state[0]["state"]["state"] = new_state_state
        new_state[0]["state"]["inc"] = new_state_inc

        rng.bit_generator.__setstate__(new_state)
