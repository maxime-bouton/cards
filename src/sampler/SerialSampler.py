from models import BaseModel
from DataManager.DataManager import DataManager
from estimator.estimatorBuilder import BaseEstimatorBuilder

import numpy as np
import sys
import h5py

class Sampler():
    def __init__(self,
                batch_size : int,
                nb_batches : int,
                seed : int,
                file_name : str,
                save_path : str,
                model : BaseModel,
                estimator_handler : BaseEstimatorBuilder  ) -> None:
        """
        Parameters
        ----------
        batch_size : int
            _description_
        nb_batches : int
            _description_
        seed : int
            _description_
        file_name : str
            _description_
        save_path : str
            _description_
        model : BaseModel
            _description_
        estimator_handler : BaseEstimatorBuilder
            _description_
        """
        self.batch_size = batch_size
        self.nb_batches = nb_batches
        self.start_batch_num = 0

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.file_name = file_name
        self.save_path = save_path

        self.model = model
        self.estimator_handler = estimator_handler

        self.potential = np.zeros([self.batch_size])

        self.data_manager = DataManager()
        
    def sample(self):
        for batch_num in range(self.start_batch_num, self.nb_batches):

            self.estimator_handler.reset()

            for i in range(self.batch_size):
                self.model.update(self.rng)

                self.potential[i] = self.model.compute_potential()
                self.estimator_handler.aggregate_states( self.model.give_data2estimator() ) #! slow down -> bench

            self.estimator_handler.build_estimator(self.batch_size)

            #save data on disk
            full_name =  self.save_path  + self.file_name + str(batch_num) + ".h5"
            with h5py.File( full_name , 'w') as file :
                self.data_manager.save_dict( self.model.get_states(), file ) 
                self.data_manager.save_array(   self.potential, file, "potential" )
                self.data_manager.save_array(   self.estimator_handler.estimator , file, self.estimator_handler.name )
                self.data_manager.save_rng( self.rng, file)

            print("Batch", batch_num, "out of", self.nb_batches, "computed.")
            print("Potential :", self.potential[-1])


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




    