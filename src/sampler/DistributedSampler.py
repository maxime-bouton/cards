from models import BaseModel
from DataManager.DataManager import DataManager
#from estimator.estimatorBuilder import BaseEstimatorBuilder

from mpi4py import MPI
import numpy as np
import sys
import h5py

class Sampler():
    def __init__(self,
                batch_size : int,
                nb_batches : int,
                seed : int, #! use generator instead of seed
                file_name : str,
                save_path : str,
                model : BaseModel  ) -> None:
        """
        Parameters
        ----------
        batch_size : int
            Lenght of a batch.
        nb_batches : int
            Number of batches to be computed.
        seed : int
            Seed of the random number generator.
        file_name : str
            Name under wich the samples will be saved.
        save_path : str
            Path to the location where we will save the samples.
        model : BaseModel
            Model used to solve an inverse problem.
        """
        self.batch_size = batch_size
        self.nb_batches = nb_batches
        self.start_batch_num = 0

        self.rank  = MPI.COMM_WORLD.Get_rank()

        # change seed for each rank
        self.seed = seed + self.rank
        self.rng = np.random.default_rng(self.seed)

        self.file_name = file_name
        self.save_path = save_path

        self.model = model

        self.potential = np.zeros([self.batch_size]) #! reduce on root

        self.data_manager = DataManager()
        
    def sample(self):
        """sampler Main method. Call the update method of the model inside a loop and save the current state at regular intarvales.
        A partial estimator is built along the iterations.
        """
        for batch_num in range(self.start_batch_num, self.nb_batches):

            self.model.estimator_builder.reset()

            for i in range(self.batch_size):
                self.model.update(self.rng)

                self.potential[i] = self.model.compute_potential() #! reduce on root
                self.model.aggregate_states() #! slow down -> bench

            self.model.estimator_builder.build_estimator(self.batch_size)

            #save data on disk
            full_name =  self.save_path  + self.file_name + str(batch_num) + ".h5"
            with h5py.File( full_name , 'w') as file :
                self.data_manager.save_dict( self.model.get_states(), file ) 
                self.data_manager.save_array(   self.potential, file, "potential" )
                self.data_manager.save_rng( self.rng, file)

            print("Batch", batch_num, "out of", self.nb_batches, "computed.") #! print on root only
            print("Potential :", self.potential[-1])