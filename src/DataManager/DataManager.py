import h5py
import numpy as np

class DataManager():
    def save(self,data : dict , filename : str) -> None:
        with h5py.File(filename, 'w') as file:
            for key in data:
                file[key] = data[key]
            #file.close()

    def save_monitoring(self,data : np.ndarray , filename : str, name : str) -> None:
        with h5py.File(filename, 'r+') as file: #expect file to exist
            file[name] = data
