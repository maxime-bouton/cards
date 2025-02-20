import logging
from os.path import join
from time import perf_counter

import h5py
import numpy as np
import cupy as cp
import torch
from mpi4py import MPI
from tqdm import tqdm

from mcmc.DataManager.DistributedDataManager import DistributedDataManager
from mcmc.models.BaseModel import BaseDistributedModel


class MultiGpuSampler:
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
        torch.cuda.set_device(self.comm.Get_rank())

        # set random number generator on each process
        if self.rank == 0:
            ss = np.random.SeedSequence(seed)
            # spawn off nworkers child SeedSequences to pass to child processes.
            child_seed = np.array(ss.spawn(comm.Get_size()))
        else:
            child_seed = None
        self.local_seed = comm.scatter(child_seed, root=0)

        with cp.cuda.Device(self.rank):
            # self.rng = torch.Generator(device="cuda").manual_seed(self.local_seed)
            self.rng = torch.Generator(device="cuda").manual_seed(
                seed + self.rank
            )  #! unsafe seed generation
        self.seed = seed + self.rank

        self.file_name = file_name
        self.save_path = save_path

        self.model = model

        self.logger = logger

        if self.rank == 0:
            self.potential = np.zeros([self.batch_size])
        self.global_potential = 0.0

        self.cpu_time = np.zeros([self.batch_size])
        self.gpu_time = np.zeros([self.batch_size])
        self.start_gpu = cp.cuda.Event()
        self.end_gpu = cp.cuda.Event()
        self.local_batch_time = 0.0

        self.data_manager = DistributedDataManager()

    def sample(self) -> None:
        """sampler Main method. Call the update method of the model inside a loop and save the current state at regular intarvales.
        A partial estimator is built along the iterations.
        """
        if self.rank == 0:
            pbar = tqdm(total=self.nb_batches, desc="Sampling", unit="it")
            pbar.update(self.start_batch_num)
        for batch_num in range(self.start_batch_num, self.nb_batches + 1):
            batch_start = perf_counter()
            self.model.estimator_builder.reset()

            for i in range(self.batch_size):
                #! doubt on gpu mesure
                self.start_gpu.record()
                start = perf_counter()
                self.model.update(self.rng)
                end = perf_counter()
                self.end_gpu.record()
                self.end_gpu.synchronize()

                self.comm.Barrier()

                partial_potential = self.model.compute_potential()
                self.cpu_time[i] = end - start
                self.gpu_time[i] = (
                    cp.cuda.get_elapsed_time(self.start_gpu, self.end_gpu) * 1e-3
                )  #! millisecond to second

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
                #!SECTION
                self.data_manager.save_rng_torch(self.rng, self.seed, file)

                self.data_manager.save_thread_array(
                    self.gpu_time,
                    self.rank,
                    self.comm.Get_size(),
                    "computation_time",
                    file,
                )
                self.data_manager.save_thread_array(
                    self.cpu_time,
                    self.rank,
                    self.comm.Get_size(),
                    "cpu_time",
                    file,
                )
            batch_end = perf_counter()
            self.batch_local_time = batch_end - batch_start
            with h5py.File(full_name, "r+", driver="mpio", comm=self.comm) as file:
                self.data_manager.save_thread_scalar(
                    np.asarray(self.local_batch_time),
                    self.rank,
                    self.comm.Get_size(),
                    "batch_time",
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
                self.logger.info("Time: {:1.3e}".format(self.gpu_time[-1]))

    def restart():
        return NotImplemented
