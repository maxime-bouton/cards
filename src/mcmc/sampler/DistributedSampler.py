import logging
from os.path import join
from time import perf_counter

import h5py
import numpy as np
from mpi4py import MPI
from tqdm import tqdm

from mcmc.DataManager.DistributedDataManager import DistributedDataManager
from mcmc.models.BaseModel import BaseDistributedModel


class DistributedSampler:
    def __init__(
        self,
        comm: MPI.Comm,
        batch_size: int,
        nb_batches: int,
        seed: int,
        file_name: str,
        save_path: str,
        model: BaseDistributedModel,
        logger: logging.Logger | None,
    ) -> None:
        """
        Parameters
        ----------
        comm : MPI.Comm
            Communication context from ``mpi4py``.
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

        self.comm = comm
        self.rank = self.comm.Get_rank()

        # set random number generator on each process
        if self.rank == 0:
            ss = np.random.SeedSequence(seed)
            # spawn off nworkers child SeedSequences to pass to child processes.
            child_seed = np.array(ss.spawn(comm.Get_size()))
        else:
            child_seed = None
        local_seed = comm.scatter(child_seed, root=0)
        self.rng = np.random.default_rng(local_seed)

        self.file_name = file_name
        self.save_path = save_path

        self.model = model

        self.logger = logger

        if self.rank == 0:
            self.potential = np.zeros([self.batch_size])
        self.local_computation_time = np.zeros([self.batch_size])
        self.global_potential = 0.0

        self.data_manager = DistributedDataManager()

    def sample(self) -> None:
        """sampler Main method. Call the update method of the model inside a loop and save the current state at regular intarvales.
        A partial estimator is built along the iterations.
        """
        if self.rank == 0:
            pbar = tqdm(total=self.nb_batches, desc="Sampling", unit="it")
            pbar.update(self.start_batch_num)
        for batch_num in range(self.start_batch_num, self.nb_batches + 1):
            self.model.estimator_builder.reset()

            for i in range(self.batch_size):
                start = perf_counter()
                self.model.update(self.rng)
                end = perf_counter()

                self.comm.Barrier()

                partial_potential = self.model.compute_potential()
                self.local_computation_time[i] = end - start

                self.global_potential = self.comm.reduce(
                    partial_potential, MPI.SUM, root=0
                )

                if self.rank == 0:
                    self.potential[i] = self.global_potential

                self.model.aggregate_states()

            self.model.estimator_builder.build_estimator(self.batch_size)

            # save data on disk
            full_name = join(self.save_path, self.file_name + str(batch_num) + ".h5")
            with h5py.File(full_name, "w", driver="mpio", comm=self.comm) as file:
                self.data_manager.save_dict(
                    self.model.get_states(),
                    file,
                    self.model.global_sizes,
                    self.model.slices,
                )
                self.data_manager.save_rng(
                    self.rng, file, self.rank, self.comm.Get_size()
                )

                self.data_manager.save_thread_array(
                    self.local_computation_time,
                    self.rank,
                    self.comm.Get_size(),
                    "computation_time",
                    file,
                )

            if self.rank == 0:
                with h5py.File(full_name, "r+") as file:
                    self.data_manager.save_local_array(
                        self.potential, "potential", file
                    )

                pbar.update()
                self.logger.info(
                    "Batch {} out of {} computed".format(batch_num, self.nb_batches)
                )
                self.logger.info("Potential: {:1.3e}".format(self.potential[-1]))
                self.logger.info(
                    "Time: {:1.3e}".format(self.local_computation_time[-1])
                )

    def restart(self, file_name: str, batch_restart: int, new_save_path: str) -> None:
        """Restart the sampler from a checkpoint file saved to disk.

        Parameters
        ----------
        file_name : str
            Name of the file from which checkpoint data are loaded.
        batch_restart : int
            Index of the iteration at which the checkpoint file has been saved.
        new_save_path : str
            Save path for the remaining samples.
        """
        with h5py.File(file_name, "r", driver="mpio", comm=self.comm) as file:
            data = self.data_manager.load_h5(file, self.model.slices)
            self.data_manager.load_rng(self.rng, file, self.rank)

        self.save_path = new_save_path
        self.start_batch_num = batch_restart + 1
        self.model.set_states(data)

        partial_potential = self.model.compute_potential()
        potential = self.comm.reduce(partial_potential, MPI.SUM, root=0)

        if self.rank == 0:
            self.logger.info(
                "Potential after restart from batch {}: {:1.3e}".format(
                    batch_restart, potential
                )
            )
