from models import BaseModel
from DataManager.DataManager import DataManager

import numpy as np

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

            self.data_manager.save( self.model.get_states(), self.save_path  + self.file_name + str(batch_num) + ".h5")
            print("Batch", batch_num, "out of", self.nb_batches, "computed.")

            self.model.reset_estimator()


    