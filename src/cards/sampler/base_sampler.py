r"""Abstract implementation of an MCMC algorithm."""

# TODO: group self.batch_size, self.nb_batches, self.file_name, self.save_path into a SamplerParameters object

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from cards.models import base_model


@dataclass
class SamplerParameters:
    r"""Dataclass gathering all parameters required to run a MCMC sampler
    :class:`cards.base_sampler.BaseSampler`.

    Attributes
    ----------
    batch_size: int
        Number of samples used to compute batched estimators to be saved in a
        checkpoint file.
    nb_batches: int
        Number of sample batches to generate before stopping the chain.
    seed: int
        Seed to initialize the random number generator.
    file_name: str
        Root name under which sample checkpoint files will be saved while
        running the chain. These can be used to restart the chain at an earlier
        state.
    save_path: str
        Path to the location where the checkpoint files are saved.
    compute_ci : bool, optional
        Boolean to trigger computation of credibility intervals, by default ``False``.
    reloaded_checkpoint : int, option
        Identifier of the checkpoint file to be loaded, by default ``0``.
    reloaded_path : str, optional
        Path to the checkpoint file to be loaded, by default ``""``.
    """

    batch_size: int
    nb_batches: int
    seed: int
    file_name: str
    save_path: str
    save_all: bool = False
    compute_ci: bool = False
    reloaded_checkpoint: int = 0
    reloaded_path: str = ""


class BaseSampler(ABC):
    """Abstract sampler implementation.

    Attributes
    ----------
    params : SamplerParameters
        Application-agnostic algorithm parameters required to run a generic
        sampler.
    model : BaseModel
        Model encoding the specific posterior distribution to be sampled.
    logger : logging.Logger
        Logger object recording the progress of the sampler.
    start_batch_num
        ...
    rank
        ...
    seed
        ...
    rng
        ...
    potential
        ...
    computation_time
        ...
    batch_time
        ...
    data_manager
        ...
    _save_full_batch

    Methods
    -------

    _make_generator(int)
        ...
    _initialize_rank()
        ...
    _get_potential() -> float
        ...
    _save_all_data(int) -> None
        ...
    restart(str, int, str) -> None
        ...
    time_mesure_begin()
        ...
    time_mesure_end()
        ...
    get_elapsed_time()
        ...
    """

    @abstractmethod
    def _make_generator(self, seed: int):
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
        save_full_batch: bool = False,
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
        self._save_full_batch = save_full_batch

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

    def sample(self) -> None:
        r"""Main iteration loop of the MCMC algorithm

        Main iteration loop in the sampler. At each iteration, calls the update
        step of the model. The current state of the parameters is regularly
        saved in checkpoint files, along with quantities required for a batched
        evaluation of the final estimates.
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

                if self.data_manager._save_full_batch:
                    self.data_manager.store_states(self.model.get_states4batch(), i)

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
