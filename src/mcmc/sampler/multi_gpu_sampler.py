import hashlib
from os.path import join

import cupy as cp
import h5py
import torch
from mpi4py import MPI

from mcmc.sampler.base_sampler import BaseSampler, SamplerParameters


class MultiGpuSampler(BaseSampler):
    def _make_generator(self, seed):
        combined = f"{self.gpu_id}{seed}"
        local_seed = int(hashlib.sha256(combined.encode()).hexdigest(), 16) % (2**32)
        return torch.Generator(device=f"cuda:{self.gpu_id}").manual_seed(local_seed)

    def _initialize_rank(self):
        return self.comm.Get_rank()

    def __init__(
        self,
        comm: MPI.Comm,
        params: SamplerParameters,
        model,
        logger,
    ):
        self.comm = comm
        self.nb_gpu = cp.cuda.runtime.getDeviceCount()
        self.gpu_id = cp.cuda.Device().id

        super().__init__(params, model, logger)

        self.start_gpu = cp.cuda.Event()
        self.end_gpu = cp.cuda.Event()

    def _time_measure_begin(self):
        self.start_gpu.record()

    def _time_measure_end(self):
        self.end_gpu.record()

    def _get_elapsed_time(self):
        # converted from milisecond to second
        return cp.cuda.get_elapsed_time(self.start_gpu, self.end_gpu) * 1e-3

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

            self.data_manager.save_rng_torch(
                self.rng, self.seed, file, self.gpu_id, self.nb_gpu
            )

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
        """Run the sampler for the specified number of batches."""
        self.model.set_slices()
        self.model.set_global_sizes()
        super().sample()

    def restart(self):
        """Restart the sampler from a checkpoint file saved to disk."""
        with h5py.File(self.reloaded_path, "r", driver="mpio", comm=self.comm) as file:
            data = self.data_manager.load_h5(file, self.model.slices)
            self.data_manager.load_rng_torch(self.rng, file, self.gpu_id)

        self.model.set_states(data)

        partial_potential = self.model.compute_potential()
        potential = self.comm.reduce(partial_potential, MPI.SUM, root=0)

        if self.rank == 0:
            self.logger.info(
                "Potential after restart from batch {}: {:1.3e}".format(
                    self.start_batch_num - 1, potential
                )
            )
