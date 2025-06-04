import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from mcmc.models import base_model


@dataclass
class SamplerParameters:
    batch_size: int
    nb_batches: int
    seed: int
    file_name: str
    save_path: str


class BaseSampler(ABC):
    @abstractmethod
    def _make_generator(self, seed: int):
        pass

    @abstractmethod
    def _make_data_manager(self):
        pass

    @abstractmethod
    def _initialize_rank(self):
        pass

    @abstractmethod
    def _get_potential(self) -> float:
        pass

    @abstractmethod
    def _save_all_data(self, batch_num: int) -> None:
        pass

    @abstractmethod
    def restart(self, file_name: str, batch_restart: int, new_save_path: str) -> None:
        pass

    @abstractmethod
    def time_mesure_begin(self):
        pass

    @abstractmethod
    def time_mesure_end(self):
        pass

    @abstractmethod
    def get_elapsed_time(self):
        pass

    def __init__(
        self,
        params: SamplerParameters,
        model: base_model,
        logger: logging.Logger | None,
    ) -> None:
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
                file_name : str
                    Name under wich the samples will be saved.
                save_path : str
                    Path to the location where we will save the samples.
        model : BaseModel
            Model used to solve an inverse problem.
        logger : logging.Logger
            Logger object.
        """
        self.batch_size = params.batch_size
        self.nb_batches = params.nb_batches
        self.start_batch_num = 1

        self.rank = self._initialize_rank()

        self.seed = params.seed
        self.rng = self._make_generator(self.seed)

        self.file_name = params.file_name
        self.save_path = params.save_path

        self.model = model

        self.logger = logger

        if self.rank == 0:
            self.potential = np.zeros([self.batch_size])
        self.computation_time = np.zeros([self.batch_size])
        self.batch_time = 0.0

        self.data_manager = self._make_data_manager()

    def sample(self) -> None:
        r"""sampler Main method. Call the update method of the model inside a loop and save the current state at regular intarvales.
        A partial estimator is built along the iterations.
        """
        if self.rank == 0:
            pbar = tqdm(total=self.nb_batches, desc="Sampling", unit="it")
            pbar.update(self.start_batch_num)

        for batch_num in range(self.start_batch_num, self.nb_batches + 1):
            self.model.estimator_builder.reset()

            for i in range(self.batch_size):
                self.time_mesure_begin()
                self.model.update(self.rng)
                self.time_mesure_end()

                global_potential = self._get_potential()
                if self.rank == 0:
                    self.potential[i] = global_potential

                self.model.aggregate_states()

                self.computation_time[i] = self.get_elapsed_time()

            self.model.estimator_builder.build_estimator(self.batch_size)

            # save data on disk
            self._save_all_data(batch_num)

            if self.rank == 0:
                pbar.update()
                self.logger.info(
                    "Batch {} out of {} computed".format(batch_num, self.nb_batches)
                )
                self.logger.info("Potential: {:1.3e}".format(self.potential[-1]))
                self.logger.info("Time: {:1.3e}".format(self.computation_time[-1]))
