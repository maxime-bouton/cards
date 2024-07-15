""" 
"""

import h5py
import numpy as np
import sys

class DataManager():
    def save_dict(self,data : dict , file ) -> None:
        for key in data:
            file[key] = data[key]

    def load(self,data : dict , file) -> None:
        for key in data:
            data[key] = file[key][:]
            #! to be checked
            #! not working as intended
    
    def load_h52dict( self, file_name : str ) -> dict :
        data = {}
        with h5py.File(file_name, 'r') as file:
            for key in file.keys():
                data[key] = file[key][:]
        return data


    def save_array(self,data : np.ndarray , file, name : str) -> None:
        file[name] = data

    def save_rng(self, rng, file) -> None :
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
        #! must be tested
