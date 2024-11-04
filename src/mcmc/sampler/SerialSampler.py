from mcmc.models import BaseModel
from mcmc.DataManager.DataManager import DataManager

import numpy as np
import sys
import h5py
from time import perf_counter
from os.path import join


class Sampler:
    def __init__(
        self,
        batch_size: int,
        nb_batches: int,
        seed: int,  #! use generator instead of seed
        file_name: str,
        save_path: str,
        model: BaseModel,
    ) -> None:
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
        self.rng = np.random.default_rng(self.seed)

        self.file_name = file_name
        self.save_path = save_path

        self.model = model

        self.potential = np.zeros([self.batch_size])
        self.computation_time = np.zeros([self.batch_size])

        self.data_manager = DataManager()

    def sample(self):
        """sampler Main method. Call the update method of the model inside a loop and save the current state at regular intarvales.
        A partial estimator is built along the iterations.
        """
        for batch_num in range(self.start_batch_num, self.nb_batches):
            self.model.estimator_builder.reset()

            for i in range(self.batch_size):
                start = perf_counter()
                self.model.update(self.rng)
                end = perf_counter()

                self.potential[i] = self.model.compute_potential()
                self.model.aggregate_states()  #! slow down -> bench

                self.computation_time[i] = end - start

            self.model.estimator_builder.build_estimator(self.batch_size)

            # save data on disk
            full_name = join(self.save_path, self.file_name + str(batch_num) + ".h5")
            with h5py.File(full_name, "w") as file:
                self.data_manager.save_dict(self.model.get_states(), file)
                self.data_manager.save_array(self.potential, file, "potential")
                self.data_manager.save_rng(self.rng, file)
                self.data_manager.save_array(
                    self.computation_time, file, "computation_time"
                )

            print("Batch", batch_num + 1, "out of", self.nb_batches, "computed.")
            print("Potential :", self.potential[-1])
            print("Time :", self.computation_time[-1])

    def set_rng(self, state_array: np.ndarray, inc_array: np.ndarray) -> None:
        """set_rng Set the internal state of the random number generator to the the one given in entry.

        Parameters
        ----------
        state_array : np.ndarray
            Internal state of the random number generator.
        inc_array : np.ndarray
            Internal state of the random number generator.
        """
        new_state = int.from_bytes(state_array, sys.byteorder)
        new_inc = int.from_bytes(inc_array, sys.byteorder)

        new_rng_state = self.rng.bit_generator.__getstate__()
        new_rng_state[0]["state"]["state"] = new_state
        new_rng_state[0]["state"]["inc"] = new_inc

        self.rng.bit_generator.__setstate__(new_rng_state)

    def restart(self, file_name: str, batch_restart: int, new_save_path: str):
        """restart Resume the sampling at a given state. It may be used to start a second where a first run had been interrupted.
        This second run will generate the exact same data that the first run would have.
        It must be called after the constructor.

        Parameters
        ----------
        file_name : str
            Name under wich the samples will be saved.
        batch_restart : int
            Number of the batch where we will resume the sampling.
        new_save_path : str
            Path to the location where we will save the samples.
        """

        data = self.data_manager.load_h5(file_name)
        self.set_rng(data["rng_state_array"], data["rng_inc_array"])
        self.model.set_states(data)

        self.save_path = new_save_path
        self.start_batch_num = batch_restart
