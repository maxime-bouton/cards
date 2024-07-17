"""
    Object than handles any reading/writing on disk.
"""

import h5py
import numpy as np
import sys

class DataManager():
    def save_dict(self,data : dict , file ) -> None:
        """save_dict Saves the dictionnary given in entry in the .h5 file given in entry. 

        Parameters
        ----------
        data : dict
            Dictionnary containing data to write on disk.
        file : _type_
            File on wich the data will be written.
        """
        for key in data:
            file[key] = data[key]
   
    def load_h5( self, file_name : str ) -> dict :
        """load_h5 Read a .h5 file and return its content in the form of a dictionnary.

        Parameters
        ----------
        file_name : str
            Full path to the file to read.

        Returns
        -------
        dict
            Dictionnary containing the data of the file.
        """
        data = {}
        with h5py.File(file_name, 'r') as file:
            for key in file.keys():
                data[key] = file[key][:]
        return data


    def save_array(self,data : np.ndarray , file, name : str) -> None:
        """save_array Save the array given in entry in the .h5 file given in entry.

        Parameters
        ----------
        data : np.ndarray
            Array of data to write on file.
        file : _type_
            File on wich we write the data.
        name : str
            Name of the datafield in the file.
        """
        file[name] = data

    def save_rng(self, rng, file) -> None :
        """save_rng Save the internal state of the random number generator given in entry to the .h5 file given in entry.

        Parameters
        ----------
        rng : _type_
            Random number generator.
        file : _type_
            File to write the internal state.
        """
        state_array = np.array( bytearray(rng.__getstate__()["state"]["state"].to_bytes(32, sys.byteorder)) )
        inc_array = np.array( bytearray(rng.__getstate__()["state"]["inc"].to_bytes(32, sys.byteorder)) )

        file["rng_state_array"] = state_array
        file["rng_inc_array"] = inc_array
        #! check saved as uint8

    # becomes useless
    def load_rng(self, rng, file_name : str) -> None:
        with h5py.File(file_name, 'r') as file:
            loaded_state_array = file["rng_state_array"][:]
            loaded_inc_array = file["rng_inc_array"][:]

        loaded_state = int.from_bytes(loaded_state_array, sys.byteorder)
        loaded_inc = int.from_bytes(loaded_inc_array, sys.byteorder)

        new_rng_state = rng.__getstate__()
        new_rng_state["state"]["state"] = loaded_state
        new_rng_state["state"]["inc"] = loaded_inc

        rng.__setstate__(new_rng_state)
