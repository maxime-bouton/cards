"""Implementation of a serial sampler."""

# TODO: documentation

from os.path import join
from time import perf_counter

import h5py
import numpy as np

from cards.data_manager.data_manager import DataManager
from cards.sampler.base_sampler import BaseSampler, SamplerParameters


class SerialSampler(BaseSampler):
    def _make_generator(self, seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    def _initialize_rank(self):
        return 0

    def _get_potential(self):
        return self.model.compute_potential()

    def time_mesure_begin(self):
        self.step_start = perf_counter()

    def time_mesure_end(self):
        self.step_end = perf_counter()

    def get_elapsed_time(self):
        return self.step_end - self.step_start

    def _save_all_data(self, batch_num):
        full_name = join(self.save_path, self.file_name + str(batch_num) + ".h5")
        with h5py.File(full_name, "w") as file:
            self.data_manager.save_dict(self.model.get_states(), file)
            self.data_manager.save_array(self.potential, file, "potential")
            self.data_manager.save_rng(self.rng, file)
            self.data_manager.save_array(
                self.computation_time, file, "computation_time"
            )
            if self.data_manager._save_full_batch:
                self.data_manager.save_batch(file)

    def __init__(self, params: SamplerParameters, model, logger, save_full_batch=False):
        super().__init__(params, model, logger, save_full_batch)

        if self._save_full_batch:
            self.data_manager = DataManager(
                self.batch_size, self._save_full_batch, self.model.get_batch_sizes()
            )
        else:
            self.data_manager = DataManager()

        self.step_start = perf_counter()
        self.step_end = perf_counter()

    def restart(self, file_name, batch_restart, new_save_path):
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
            self.data_manager.load_rng(self.rng, file)

        self.save_path = new_save_path
        self.start_batch_num = batch_restart + 1
        self.model.set_states(data)

        potential = self._get_potential()
        self.logger.info(
            "Potential after restart from batch {}: {:1.3e}".format(
                batch_restart, potential
            )
        )
