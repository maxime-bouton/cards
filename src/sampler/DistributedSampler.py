from models.BaseModel import BaseDistributedModel
from DataManager.DistributedDataManager import DistributedDataManager
#from estimator.estimatorBuilder import BaseEstimatorBuilder

from mpi4py import MPI
import numpy as np
import sys
import h5py

class DistributedSampler():
    def __init__(self,
                batch_size : int,
                nb_batches : int,
                seed : int, 
                file_name : str,
                save_path : str,
                model : BaseDistributedModel ) -> None:
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

        self.potential = np.zeros([self.batch_size]) #! memory management, useless exepct on root

        self.data_manager = DistributedDataManager()
        
    def sample(self):
        """sampler Main method. Call the update method of the model inside a loop and save the current state at regular intarvales.
        A partial estimator is built along the iterations.
        """
        for batch_num in range(self.start_batch_num, self.nb_batches):

            self.model.estimator_builder.reset()

            for i in range(self.batch_size):

                self.model.update(self.rng)
                MPI.COMM_WORLD.Barrier()

                partial_potential = self.model.compute_potential()
                
                MPI.COMM_WORLD.reduce(partial_potential, MPI.SUM, 0)
                self.potential[i] = partial_potential # sum -> full potential
        
                self.model.aggregate_states()

            self.model.estimator_builder.build_estimator(self.batch_size)
            
            #print(self.rank, self.model.slices["Z"]) #! error here
            #save data on disk
            full_name =  self.save_path  + self.file_name + str(batch_num) + ".h5"
            with h5py.File( full_name , 'w', driver='mpio', comm=MPI.COMM_WORLD) as file :
                self.data_manager.save_dict( self.model.get_states(), file, self.model.global_sizes, self.model.slices ) 
                #self.data_manager.save_array(   self.potential, file, "potential" )
                #self.data_manager.save_rng( self.rng, file)

            if self.rank ==0 :
                print("Batch", batch_num, "out of", self.nb_batches, "computed.") #! print on root only
                print("Potential :", self.potential[-1])