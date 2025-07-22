from os.path import join
from time import perf_counter

import h5py
import numpy as np
from mpi4py import MPI

from mcmc.data_manager.data_manager import DataManager
from mcmc.sampler.base_sampler import BaseSampler, SamplerParameters


class DistributedSampler(BaseSampler):
    """DistributedSampler Sampler to use for distributed models, computations must be done on cpu.

    Parameters
    ----------
    BaseSampler : _type_
        _description_
    """

    def _make_generator(self, seed: int):
        # set random number generator on each process
        if self.rank == 0:
            ss = np.random.SeedSequence(seed)
            # spawn off nworkers child SeedSequences to pass to child processes.
            child_seed = np.array(ss.spawn(self.comm.Get_size()))
        else:
            child_seed = None
        local_seed = self.comm.scatter(child_seed, root=0)
        return np.random.default_rng(local_seed)

    def _initialize_rank(self):
        return self.comm.Get_rank()

    def __init__(
        self,
        comm: MPI.Comm,
        params: SamplerParameters,
        model,
        logger,
        save_full_batch=False,
    ):
        self.comm = comm

        super().__init__(params, model, logger, save_full_batch)

        if self._save_full_batch:
            self.data_manager = DataManager(
                self.batch_size,
                save_full_batch,
                self.model.get_batch_sizes(),
                self.model.global_sizes,
                self.model.slices,
            )
        else:
            self.data_manager = DataManager()

        self.step_start = perf_counter()
        self.step_end = perf_counter()

    def time_mesure_begin(self):
        self.step_start = perf_counter()

    def time_mesure_end(self):
        self.step_end = perf_counter()

    def get_elapsed_time(self):
        return self.step_end - self.step_start

    def _get_potential(self):
        partial_potential = self.model.compute_potential()
        global_potential = self.comm.reduce(partial_potential, MPI.SUM, root=0)
        return global_potential

    def _save_all_data(self, batch_num: int) -> None:
        full_name = join(self.save_path, self.file_name + str(batch_num) + ".h5")
        with h5py.File(full_name, "w", driver="mpio", comm=self.comm) as file:
            self.data_manager.save_dict(
                self.model.get_states(),
                file,
                self.model.global_sizes,
                self.model.slices,
            )
            self.data_manager.save_rng(self.rng, file, self.rank, self.comm.Get_size())

            self.data_manager.save_thread_array(
                self.computation_time,
                self.rank,
                self.comm.Get_size(),
                "computation_time",
                file,
            )

            if self.data_manager._save_full_batch:
                self.data_manager.save_batch(file)

        if self.rank == 0:
            with h5py.File(full_name, "r+") as file:
                self.data_manager.save_local_array(self.potential, "potential", file)

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
            data = self.data_manager.load_h5(
                file, self.model.local_sizes, self.model.slices
            )
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
