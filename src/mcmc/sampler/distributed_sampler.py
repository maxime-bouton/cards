import logging
from os.path import join
from time import perf_counter

import h5py
import numpy as np
from mpi4py import MPI

from mcmc.models.base_model import BaseDistributedModel
from mcmc.sampler.base_sampler import BaseSampler, SamplerParameters


class DistributedSampler(BaseSampler):
    """DistributedSampler Sampler to use for distributed models, computations must be done on cpu.

    Parameters
    ----------
    BaseSampler : _type_
        _description_
    """

    def _make_generator(self, seed: int) -> np.random.Generator:
        # set random number generator on each process
        if self.rank == 0:
            ss = np.random.SeedSequence(seed)
            # spawn off nworkers child SeedSequences to pass to child processes.
            child_seeds = ss.spawn(self.comm.Get_size())
        else:
            child_seeds = None
        local_seed = self.comm.scatter(child_seeds, root=0)
        return np.random.default_rng(local_seed)

    def _initialize_rank(self) -> int:
        return self.comm.Get_rank()

    def __init__(
        self,
        comm: MPI.Comm,
        params: SamplerParameters,
        model: BaseDistributedModel,
        logger: logging.Logger | None,
    ):
        self.comm = comm

        super().__init__(params, model, logger)

        self.step_start = perf_counter()
        self.step_end = perf_counter()

    def _time_measure_begin(self):
        self.step_start = perf_counter()

    def _time_measure_end(self):
        self.step_end = perf_counter()

    def _get_elapsed_time(self) -> float:
        return self.step_end - self.step_start

    def _get_potential(self) -> float:
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
        if self.rank == 0:
            with h5py.File(full_name, "r+") as file:
                self.data_manager.save_local_array(self.potential, "potential", file)

    def sample(self):
        self.model.set_slices()
        self.model.set_global_sizes()
        super().sample()

    def restart(self):
        """Restart the sampler from a checkpoint file saved to disk."""
        with h5py.File(self.reloaded_path, "r", driver="mpio", comm=self.comm) as file:
            data = self.data_manager.load_h5(file, self.model.slices)
            self.data_manager.load_rng(self.rng, file, self.rank)

        self.model.set_states(data)

        partial_potential = self.model.compute_potential()
        potential = self.comm.reduce(partial_potential, MPI.SUM, root=0)

        if self.rank == 0:
            self.logger.info(
                "Potential after restart from batch {}: {:1.3e}".format(
                    self.start_batch_num - 1, potential
                )
            )
