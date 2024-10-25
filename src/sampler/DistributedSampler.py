from models.BaseModel import BaseDistributedModel
from DataManager.DistributedDataManager import DistributedDataManager
# from estimator.estimatorBuilder import BaseEstimatorBuilder

from mpi4py import MPI
import numpy as np
import h5py
from time import perf_counter
from os.path import join


class DistributedSampler:
    def __init__(
        self,
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

        self.rank = MPI.COMM_WORLD.Get_rank()

        # change seed for each rank
        self.seed = [
            seed,
            self.rank,
        ]  #! seed generation not safe : https://numpy.org/doc/stable/reference/random/parallel.html#seedsequence-spawn
        self.rng = np.random.default_rng(self.seed)

        self.file_name = file_name
        self.save_path = save_path

        self.model = model

        self.potential = np.zeros(
            [self.batch_size]
        )  #! memory management, useless excepted on root
        self.computation_time = np.zeros([self.batch_size])

        self.data_manager = DistributedDataManager()

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

                MPI.COMM_WORLD.Barrier()

                partial_potential = self.model.compute_potential()
                elapsed = end - start

                self.potential[i] = MPI.COMM_WORLD.reduce(
                    partial_potential, MPI.SUM, root=0
                )  # sum -> full potential
                self.computation_time[i] = MPI.COMM_WORLD.reduce(
                    elapsed, MPI.MAX, root=0
                )

                self.model.aggregate_states()

            self.model.estimator_builder.build_estimator(self.batch_size)

            # save data on disk
            full_name = join( self.save_path  , self.file_name + str(batch_num) + ".h5")
            with h5py.File(full_name, "w", driver="mpio", comm=MPI.COMM_WORLD) as file:
                self.data_manager.save_dict(
                    self.model.get_states(),
                    file,
                    self.model.global_sizes,
                    self.model.slices,
                )
                self.data_manager.save_rng(
                    self.rng, file, self.rank, MPI.COMM_WORLD.Get_size()
                )

            if self.rank == 0:
                with h5py.File(full_name, "r+") as file:
                    self.data_manager.save_local_array(
                        self.potential, "potential", file
                    )
                    self.data_manager.save_local_array(
                        self.computation_time, "computation_time", file
                    )

            if self.rank == 0:
                print(
                    "Batch", batch_num, "out of", self.nb_batches, "computed."
                )  #! move to logger
                print("Potential :", self.potential[-1])
                print("Time :", self.computation_time[-1])

    def restart(self, file_path: str, new_save_path: str, batch_restart: int) -> None:
        with h5py.File(file_path, "r", driver="mpio", comm=MPI.COMM_WORLD) as file:
            data = self.data_manager.load_h5(file, self.model.slices)
            self.data_manager.load_rng(self.rng, file, self.rank)

        self.save_path = new_save_path
        self.start_batch_num = batch_restart
        self.model.set_states(data)
