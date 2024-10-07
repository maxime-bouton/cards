"""
    Object than handles any reading/writing on disk with parallel memory acces.
"""

import h5py
import numpy as np

class DistributedDataManager():
    def save_dict(self,data : dict , file : h5py.File, global_sizes : dict ,slices : dict ) -> None:
        """save_dict Saves the dictionnary given in entry in the .h5 file given in entry. 

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
            dset = file.create_dataset( key, global_sizes[key], dtype='f') #! must give global dimensions
            dset[ slices[key] ] = data[key]


    def save_array(self,data : np.ndarray , global_size : np.ndarray ,slices : slice , file : h5py.File, name : str) -> None:
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
        dset = file.create_dataset( name, global_size, dtype='f')
        dset[*slices] = data
