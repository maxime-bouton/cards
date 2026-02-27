import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from mcmc.data_manager.data_manager import DataManager
from mcmc.models.base_model import BaseModel


@dataclass
class SamplerParameters:
    batch_size: int
    nb_batches: int
    seed: int
    save_path: str
    save_all: bool
    compute_ci: bool
    reloaded_checkpoint: int = 0
    reloaded_path: str = ""


class BaseSampler(ABC):
    @abstractmethod
    def _make_generator(self, seed: int) -> np.random.Generator | torch.Generator: ...

    @abstractmethod
    def _initialize_rank(self) -> int: ...

    @abstractmethod
    def _get_potential(self) -> float: ...

    @abstractmethod
    def _save_all_data(self, batch_num: int): ...

    @abstractmethod
    def restart(self): ...

    @abstractmethod
    def _time_measure_begin(self): ...

    @abstractmethod
    def _time_measure_end(self): ...

    @abstractmethod
    def _get_elapsed_time(self) -> float: ...

    def __init__(
        self,
        params: SamplerParameters,
        model: BaseModel,
        logger: logging.Logger | None,
    ):
        """
        Parameters
        ----------
        params : SamplerParameters
            Dataclass containing :
                batch_size : int
                    Lenght of a batch.
                nb_batches : int
                    Number of batches to be computed.
                seed : int
                    Seed of the random number generator.
                save_path : str
                    Path to the location where we will save the samples.
        model : BaseModel
            Model used to solve an inverse problem.
        logger : logging.Logger
            Logger object.
        """
        self.batch_size = params.batch_size
        self.nb_batches = params.nb_batches

        self.rank = self._initialize_rank()

        self.seed = params.seed
        self.rng = self._make_generator(self.seed)

        self.file_name = "checkpoint_"
        self.save_path = params.save_path

        self.model = model

        if params.save_all or params.compute_ci:
            self.model.estimator_builder.set_batch_size(self.batch_size)
            if params.save_all:
                self.model.estimator_builder.enable_save_all()
            if params.compute_ci:
                self.model.estimator_builder.enable_compute_ci()

        self.logger = logger

        if self.rank == 0:
            self.potential = np.zeros([self.batch_size])
        self.computation_time = np.zeros([self.batch_size])
        self.batch_time = 0.0

        self.data_manager = DataManager()

        if params.reloaded_checkpoint > 0:
            self.reloaded_path = params.reloaded_path
            self.start_batch_num = params.reloaded_checkpoint + 1
            self.restart()
        else:
            self.start_batch_num = 1

    def sample(self):
        """sampler Main method. Call the update method of the model inside a loop and save the current state at regular intarvales.
        A partial estimator is built along the iterations.
        """
        if self.rank == 0:
            pbar = tqdm(total=self.nb_batches, desc="Sampling", unit="it")

        for batch_num in range(self.start_batch_num, self.nb_batches + 1):
            self.model.estimator_builder.reset()

            for i in range(self.batch_size):
                self.logger.info("Batch {} iteration {}".format(batch_num, i))
                self._time_measure_begin()
                self.model.update(self.rng)
                self._time_measure_end()

                global_potential = self._get_potential()
                if self.rank == 0:
                    self.potential[i] = global_potential

                self.model.aggregate_states()

                self.computation_time[i] = self._get_elapsed_time()

            self.model.estimator_builder.build_estimator()

            # save data on disk
            self._save_all_data(batch_num)

            if self.rank == 0:
                pbar.update()
                self.logger.info(
                    "Batch {} out of {} computed".format(batch_num, self.nb_batches)
                )
                self.logger.info("Potential: {:1.3e}".format(self.potential[-1]))
                self.logger.info("Time: {:1.3e}".format(self.computation_time[-1]))
