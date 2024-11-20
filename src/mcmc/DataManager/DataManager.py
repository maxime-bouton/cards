"""
Object than handles any reading/writing on disk.
"""

import h5py
import numpy as np
import torch

from mcmc.DataManager.warmstart_rng import (
    load_rng_np,
    save_rng_np,
    save_rng_offset_torch,
    load_rng_offset_torch,
)


# FIXME: this class will need to be adapted to accommodate torch array


class DataManager:
    def __init__(self):
        pass

    def save_dict(self, data: dict, file: h5py.File) -> None:
        """save_dict Saves the dictionnary given in entry in the .h5 file given in entry.

        Parameters
        ----------
        data : dict
            Dictionnary containing data to write on disk.
        file : h5py.File
            File on wich the data will be written.
        """
        for key in data:
            file[key] = data[key]

    def load_h5(self, file: h5py.File) -> dict:
        """load_h5 Read a .h5 file and return its content in the form of a dictionary.

        Parameters
        ----------
        file : h5py.File
            File from which the data is read.

        Returns
        -------
        dict
            Dictionnary containing the data of the file.
        """
        data = {}
        for key in file.keys():
            if file[key].size > 1:
                data[key] = file[key][:]
            else:
                data[key] = file[key][()]
        return data

    def save_array(self, data: np.ndarray, file: h5py.File, name: str) -> None:
        """save_array Save the array given in entry in the .h5 file given in entry.

        Parameters
        ----------
        data : np.ndarray
            Array of data to write on file.
        file : h5py.File
            File on wich we write the data.
        name : str
            Name of the datafield in the file.
        """
        file[name] = data

    def save_rng(self, rng: np.random.Generator, file: h5py.File) -> None:
        """save_rng Save the internal state of a numpy random number generator
        into a .h5 file.

        Parameters
        ----------
        rng : np.random.Generator
            Numpy random number generator.
        file : h5py.File
            Handle to a `.h5` file to save the state of the generator.
        """
        save_rng_np(rng, file)
        return

    def load_rng(self, rng: np.random.Generator, file: h5py.File) -> None:
        """load_rng Load the state of a numpy random number generator from a .h5 file.

        Parameters
        ----------
        rng : np.random.Generator
            Numpy random number generator.
        file : h5py.File
            Handle to a `.h5` file from which the state of the generator will be loeaded.
        """
        load_rng_np(rng, file)
        return

    def save_rng_torch(
        self, rng: torch._C.Generator, seed: int, h5file: h5py.File
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

        Note
        ----
        Requires ``pythorch>=2.5``. Only supported for generators on the GPU.
        """
        save_rng_offset_torch(rng, seed, h5file)
        return

    def load_rng_torch(self, rng: torch._C.Generator, h5file: h5py.File):
        r"""Load the state of a pytorch random number generator from a .h5 file using the offset from an initial seed state.

        Parameters
        ----------
        rng : torch._C.Generator
            Pytorch random number generator.
        h5file : h5py.File
            Handle to a `.h5` file to save the state of the generator.

        Note
        ----
        Requires ``pythorch>=2.5``. Only supported for generators on the GPU.
        """
        load_rng_offset_torch(rng, h5file)
        return
