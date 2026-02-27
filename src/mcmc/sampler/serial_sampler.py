from os.path import join
from time import perf_counter

import h5py
import numpy as np

from mcmc.sampler.base_sampler import BaseSampler, SamplerParameters


class SerialSampler(BaseSampler):
    def __init__(self, params: SamplerParameters, model, logger):
        super().__init__(params, model, logger)

        self.step_start = perf_counter()
        self.step_end = perf_counter()

    def restart(self):
        """Resume the sampling at a given state. It may be used to start a second where a first run had been interrupted.
        This second run will generate the exact same data that the first run would have.
        It must be called after the constructor.
        """
        with h5py.File(self.reloaded_path, "r") as file:
            data = self.data_manager.load_h5(file)
            self.data_manager.load_rng(self.rng, file)

        self.model.set_states(data)

        potential = self._get_potential()
        self.logger.info(
            "Potential after restart from batch {}: {:1.3e}".format(
                self.start_batch_num - 1, potential
            )
        )

    def _time_measure_begin(self):
        self.step_start = perf_counter()

    def _time_measure_end(self):
        self.step_end = perf_counter()

    def _get_elapsed_time(self):
        return self.step_end - self.step_start

    def _get_potential(self):
        return self.model.compute_potential()

    def _initialize_rank(self):
        return 0

    def _make_generator(self, seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    def _save_all_data(self, batch_num):
        full_name = join(self.save_path, self.file_name + str(batch_num) + ".h5")
        with h5py.File(full_name, "w") as file:
            self.data_manager.save_dict(self.model.get_states(), file)
            self.data_manager.save_array(self.potential, file, "potential")
            self.data_manager.save_rng(self.rng, file)
            self.data_manager.save_array(
                self.computation_time, file, "computation_time"
            )
