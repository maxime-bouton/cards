"""Implementation of a sampler using buffers on GPU."""

from os.path import join

import cupy as cp
import h5py
import torch

from mcmc.data_manager.data_manager import DataManager
from mcmc.sampler.base_sampler import BaseSampler, SamplerParameters


class GpuSampler(BaseSampler):
    def _make_generator(self, seed):
        return torch.Generator(device="cuda").manual_seed(seed)

    def _initialize_rank(self):
        return 0

    def time_mesure_begin(self):
        self.start_gpu.record()

    def time_mesure_end(self):
        self.end_gpu.record()

    def get_elapsed_time(self):
        return (
            cp.cuda.get_elapsed_time(self.start_gpu, self.end_gpu) * 1e-3
        )  # converted from milisecond to second

    def __init__(self, params: SamplerParameters, model, logger, save_full_batch=False):
        super().__init__(params, model, logger, save_full_batch)

        if self._save_full_batch:
            self.data_manager = DataManager(
                self.batch_size, self._save_full_batch, self.model.get_batch_sizes()
            )
        else:
            self.data_manager = DataManager()

        self.start_gpu = cp.cuda.Event()
        self.end_gpu = cp.cuda.Event()

    def _get_potential(self):
        return self.model.compute_potential()

    def _save_all_data(self, batch_num: int) -> None:
        full_name = join(self.save_path, self.file_name + str(batch_num) + ".h5")
        with h5py.File(full_name, "w") as file:
            self.data_manager.save_dict(self.model.get_states(), file)
            self.data_manager.save_array(self.potential, file, "potential")
            self.data_manager.save_array(
                self.computation_time, file, "computation_time"
            )
            self.data_manager.save_rng_torch(self.rng, self.seed, file)

            if self._save_full_batch:
                self.data_manager.save_batch(file, True)

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
