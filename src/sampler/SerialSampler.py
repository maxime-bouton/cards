from models import BaseModel
from DataManager.DataManager import DataManager

import numpy as np
import sys

class Sampler():
    def __init__(self,
                batch_size : int,
                nb_batches : int,
                seed : int,
                file_name : str,
                save_path : str,
                model : BaseModel ) -> None:
        self.batch_size = batch_size
        self.nb_batches = nb_batches
        self.start_batch_num = 0

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.file_name = file_name
        self.save_path = save_path

        self.model = model

        self.potential = np.zeros([self.batch_size])

        self.data_manager = DataManager()
        
    def sample(self):
        for batch_num in range(self.start_batch_num, self.nb_batches):
            for i in range(self.batch_size):
                self.model.update(self.rng)

                self.potential[i] = self.model.compute_potential()

            #save current states
            self.model.normalize_estimator( self.batch_size )

            full_name =  self.save_path  + self.file_name + str(batch_num) + ".h5"
            self.data_manager.save( self.model.get_states(), full_name )
            self.data_manager.save_monitoring(   self.potential, full_name, "potential" )
            self.data_manager.save_rng( self.rng, full_name)

            print("Batch", batch_num, "out of", self.nb_batches, "computed.")
            print("Potential :", self.potential[-1])

            self.model.reset_estimator()

    def set_rng(self, state_array : np.ndarray, inc_array : np.ndarray) -> None :
        new_state = int.from_bytes(state_array, sys.byteorder)
        new_inc = int.from_bytes(inc_array, sys.byteorder)

        new_rng_state = self.rng.__getstate__()
        new_rng_state["state"]["state"] = new_state
        new_rng_state["state"]["inc"] = new_inc

        self.rng.__setstate__(new_rng_state)


    def restart(self, file_name : str, batch_restart : int, new_save_path : str):
    
        #self.data_manager.load_rng( self.rng, file_name)
        #self.data_manager.load( self.model.get_states(), file_name )

        data = self.data_manager.load_h52dict( file_name)
        self.set_rng( data["rng_state_array"], data["rng_inc_array"] )
        self.model.set_states(data)

        self.save_path = new_save_path
        self.start_batch_num = batch_restart




    