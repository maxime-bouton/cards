import logging
from os.path import join
from time import perf_counter

import h5py
import numpy as np
import torch
from tqdm import tqdm

from mcmc.DataManager.DataManager import DataManager
from mcmc.models import BaseModel


class GpuSampler:
    def __init__(
        self,
        batch_size: int,
        nb_batches: int,
        seed: int,
        file_name: str,
        save_path: str,
        model: BaseModel,
        logger: logging.Logger,
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
        logger : logging.Logger
            Logger object.
        """
        self.batch_size = batch_size
        self.nb_batches = nb_batches
        self.start_batch_num = 1

        self.seed = seed
        self.rng = torch.Generator(device="cuda").manual_seed(self.seed)

        self.file_name = file_name
        self.save_path = save_path

        self.model = model

        self.logger = logger

        self.potential = np.zeros([self.batch_size])
        self.computation_time = np.zeros([self.batch_size])

        self.data_manager = DataManager()

    def sample(self):
        r"""Sampler main method. Call the update method of the model inside a loop and save the current state at regular intarvales.
        A partial estimator is built along the iterations.
        """
        pbar = tqdm(total=self.nb_batches, desc="Sampling", unit="it")
        pbar.update(self.start_batch_num)
        for batch_num in range(self.start_batch_num, self.nb_batches + 1):
            self.model.estimator_builder.reset()

            for i in range(self.batch_size):
                start = perf_counter()
                self.model.update(self.rng)
                end = perf_counter()

                elapsed = end - start

                self.potential[i] = self.model.compute_potential()
                self.computation_time[i] = elapsed
                self.model.aggregate_states()

            self.model.estimator_builder.build_estimator(self.batch_size)

            # save data on disk
            full_name = join(self.save_path, self.file_name + str(batch_num) + ".h5")
            with h5py.File(full_name, "w") as file:
                self.data_manager.save_dict(self.model.get_states(), file)
                self.data_manager.save_array(self.potential, file, "potential")
                self.data_manager.save_array(
                    self.computation_time, file, "computation_time"
                )
                self.data_manager.save_rng_torch(self.rng, self.seed, file)

            pbar.update()
            self.logger.info(
                "Batch {} out of {} computed".format(batch_num, self.nb_batches)
            )
            self.logger.info("Potential: {:1.3e}".format(self.potential[-1]))
            self.logger.info("Time: {:1.3e}".format(self.computation_time[-1]))

    def restart(self, file_name: str, batch_restart: int, new_save_path: str):
        r"""restart Resume the sampling at a given state. It may be used to start a second where a first run had been interrupted.
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
        with h5py.File(file_name, "r") as file:
            data = self.data_manager.load_h5(file)
            self.data_manager.load_rng_torch(self.rng, file)

        self.save_path = new_save_path
        self.start_batch_num = batch_restart + 1
        self.model.set_states(data)

        potential = self.model.compute_potential()
        self.logger.info(
            "Potential after restart from batch {}: {:1.3e}".format(
                batch_restart, potential
            )
        )
