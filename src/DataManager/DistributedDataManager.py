"""
    Object than handles any reading/writing on disk with parallel memory acces.
"""

import h5py
import numpy as np
import sys

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

    def save_seed(self, seed : int , rank : int , comm_size : int , file : h5py.File ):
        dset = file.create_dataset("seed", np.asarray([comm_size]), dtype=int)
        dset[rank] = seed

    def save_local_array(self, data : np.ndarray, name : str , file : h5py.File):
        file[ name ] = data

    def save_rng(self, rng : np.random.Generator, file : h5py.File, rank : int, comm_size : int) -> None :
        
        state_array = np.array( bytearray(rng.bit_generator.__getstate__()[0]["state"]["state"].to_bytes(32, sys.byteorder)) )
        inc_array = np.array( bytearray(rng.bit_generator.__getstate__()[0]["state"]["inc"].to_bytes(32, sys.byteorder)) )

        size = np.asarray([comm_size, *state_array.shape])
        dset = file.create_dataset( "rng_state_array", size )
        dset[rank] = state_array

        #dset = file.create_dataset("rng_inc_state", (comm_size,*inc_array.shape))
        #dset["rng_inc_array"][rank,:] = inc_array