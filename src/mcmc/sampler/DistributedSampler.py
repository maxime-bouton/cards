from os.path import join
from time import perf_counter

import h5py
import numpy as np

# from estimator.estimatorBuilder import BaseEstimatorBuilder
from mpi4py import MPI

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

        # FIXME: memory management, useless except on root
        self.potential = np.zeros([self.batch_size])
        self.computation_time = np.zeros([self.batch_size])

        self.data_manager = DistributedDataManager()

    def sample(self) -> None:
        """sampler Main method. Call the update method of the model inside a loop and save the current state at regular intarvales.
        A partial estimator is built along the iterations.
        """
        for batch_num in range(self.start_batch_num, self.nb_batches + 1):
            self.model.estimator_builder.reset()

            for i in range(self.batch_size):
                start = perf_counter()
                self.model.update(self.rng)
                end = perf_counter()

                self.comm.Barrier()

                partial_potential = self.model.compute_potential()
                elapsed = end - start

                self.potential[i] = self.comm.reduce(
                    partial_potential, MPI.SUM, root=0
                )  # sum -> full potential
                self.computation_time[i] = self.comm.reduce(elapsed, MPI.MAX, root=0)

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

            if self.rank == 0:
                with h5py.File(full_name, "r+") as file:
                    self.data_manager.save_local_array(
                        self.potential, "potential", file
                    )
                    self.data_manager.save_local_array(
                        self.computation_time, "computation_time", file
                    )

                # FIXME move to logger
                print("Batch", batch_num, "out of", self.nb_batches, "computed.")
                print("Potential :", self.potential[-1])
                print("Time :", self.computation_time[-1])

    def restart(self, file_name: str, new_save_path: str, batch_restart: int) -> None:
        """Restart the sampler from a checkpoint file saved to disk.

        Parameters
        ----------
        file_name : str
            Name of the file from which checkpoint data are loaded.
        new_save_path : str
            Save path for the remaining samples.
        batch_restart : int
            Index of the iteration at which the checkpoint file has been saved.
        """
        with h5py.File(file_name, "r", driver="mpio", comm=self.comm) as file:
            data = self.data_manager.load_h5(file, self.model.slices)
            self.data_manager.load_rng(self.rng, file, self.rank)

        self.save_path = new_save_path
        self.start_batch_num = batch_restart + 1
        self.model.set_states(data)

        partial_potential = self.model.compute_potential()
        potential = self.comm.reduce(partial_potential, MPI.SUM, root=0)

        # FIXME move to logger
        if self.rank == 0:
            print("Potential after restart: {}".format(potential))
