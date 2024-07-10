import h5py
import numpy as np
import sys

class DataManager():
    def save(self,data : dict , file_name : str) -> None:
        with h5py.File(file_name, 'w') as file:
            for key in data:
                file[key] = data[key]

    def load(self,data : dict , file_name : str) -> None:
        with h5py.File(file_name, 'r') as file:
            for key in data:
                data[key] = file[key][:]
                #! to be checked

    def save_monitoring(self,data : np.ndarray , file_name : str, name : str) -> None:
        with h5py.File(file_name, 'r+') as file: #expect file to exist
            file[name] = data

    def save_rng(self, rng, file_name : str) -> None :
        state_array = np.array( bytearray(rng.__getstate__()["state"]["state"].to_bytes(32, sys.byteorder)) )
        inc_array = np.array( bytearray(rng.__getstate__()["state"]["inc"].to_bytes(32, sys.byteorder)) )

        with h5py.File(file_name, 'r+') as file: #expect file to exist
            file["rng_state_array"] = state_array
            file["rng_inc_array"] = inc_array
            #! check saved as uint8

    def load_rng(self, rng, file_name : str) -> None:
        with h5py.File(file_name, 'r') as file:
            loaded_state_array = file["rng_state_array"][:]
            loaded_inc_array = file["rng_inc_array"][:]

        loaded_state = int.from_bytes(loaded_state_array, sys.byteorder)
        loaded_inc = int.from_bytes(loaded_inc_array, sys.byteorder)

        current_rng_state = rng.__getstate__()
        current_rng_state["state"]["state"] = loaded_state
        current_rng_state["state"]["inc"] = loaded_inc

        rng.__setstate__(current_rng_state)
        #! must be tested
