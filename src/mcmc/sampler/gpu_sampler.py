from os.path import join

import cupy as cp
import h5py
import torch

from mcmc.sampler.base_sampler import BaseSampler, SamplerParameters


class GpuSampler(BaseSampler):
    def __init__(self, params: SamplerParameters, model, logger):
        super().__init__(params, model, logger)

        self.start_gpu = cp.cuda.Event()
        self.end_gpu = cp.cuda.Event()

    def restart(self):
        """Resume the sampling at a given state. It may be used to start a second where a first run had been interrupted.
        This second run will generate the exact same data that the first run would have.
        It must be called after the constructor.
        """
        with h5py.File(self.reloaded_path, "r") as file:
            data = self.data_manager.load_h5(file)
            self.data_manager.load_rng_torch(self.rng, file)

        self.model.set_states(data)

        potential = self.model.compute_potential()
        self.logger.info(
            "Potential after restart from batch {}: {:1.3e}".format(
                self.start_batch_num - 1, potential
            )
        )

    def _time_measure_begin(self):
        self.start_gpu.record()

    def _time_measure_end(self):
        self.end_gpu.record()

    def _get_elapsed_time(self) -> float:
        return (
            cp.cuda.get_elapsed_time(self.start_gpu, self.end_gpu) * 1e-3
        )  # converted from milisecond to second

    def _get_potential(self):
        return self.model.compute_potential()

    def _initialize_rank(self):
        return 0

    def _make_generator(self, seed):
        return torch.Generator(device="cuda").manual_seed(seed)

    def _save_all_data(self, batch_num: int) -> None:
        full_name = join(self.save_path, self.file_name + str(batch_num) + ".h5")
        with h5py.File(full_name, "w") as file:
            self.data_manager.save_dict(self.model.get_states(), file)
            self.data_manager.save_array(self.potential, file, "potential")
            self.data_manager.save_array(
                self.computation_time, file, "computation_time"
            )
            self.data_manager.save_rng_torch(self.rng, self.seed, file)
