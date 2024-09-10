from models import BaseModel
from DataManager.DataManager import DataManager
from estimator.GpuEstimatorBuilder import BaseGpuEstimatorBuilder

import torch
import cupy as cp
import numpy as np
import sys
import h5py

class GpuSampler():
    def __init__(self,
                batch_size : int,
                nb_batches : int,
                seed : int,
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

        self.seed = seed
        #self.rng = cp.random.default_rng(self.seed)
        #self.rng = cp.random.RandomState(seed = self.seed, method  = cp.cuda.curand.CURAND_RNG_PSEUDO_DEFAULT)
        # randomState is not a generator but it contains one , see : https://github.com/cupy/cupy/blob/main/cupy/random/_generator.py

        self.rng = torch.Generator(device='cuda').manual_seed(self.seed)

        self.file_name = file_name
        self.save_path = save_path

        self.model = model

        self.potential = np.zeros([self.batch_size])

        self.data_manager = DataManager()

    def sample(self):
            """Sampler main method. Call the update method of the model inside a loop and save the current state at regular intarvales.
            A partial estimator is built along the iterations.
            """
            for batch_num in range(self.start_batch_num, self.nb_batches):

                self.model.estimator_builder.reset()

                for i in range(self.batch_size):
                    self.model.update(self.rng)

                    self.potential[i] = self.model.compute_potential()
                    self.model.aggregate_states() #! slow down -> bench

                self.model.estimator_builder.build_estimator(self.batch_size)

                #save data on disk
                full_name =  self.save_path  + self.file_name + str(batch_num) + ".h5"
                with h5py.File( full_name , 'w') as file :
                    self.data_manager.save_dict( self.model.get_states(), file ) 
                    self.data_manager.save_array( self.potential, file, "potential" )
                    self.data_manager.save_offset( self.rng, file)

                print("Batch", batch_num, "out of", self.nb_batches, "computed.")
                print("Potential :", self.potential[-1])

    def restart_rng(self, offset):
        #self.rng = cp.random.RandomState(seed = self.seed, method  = cp.cuda.curand.CURAND_RNG_PSEUDO_DEFAULT)
        #offset = ( self.start_batch_num * self.batch_size  * self.model.get_step_offset() ) //2
        #cp.cuda.curand.setGeneratorOffset(self.rng._generator, offset)
        self.rng.set_offset(offset)

    def restart(self, file_name : str, batch_restart : int, new_save_path : str):

        data = self.data_manager.load_h5( file_name)
        self.save_path = new_save_path
        self.start_batch_num = batch_restart

        self.restart_rng( int( data['offset'][()] ) )
        self.model.set_states(data)   