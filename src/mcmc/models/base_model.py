r"""Abstract class that descibes the interface of the model class. The methods
declared here will be called within the sampler.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch

from mcmc.estimator.estimator_builder import EstimatorBuilder


class BaseModel(ABC):
    estimator_builder: EstimatorBuilder

    @abstractmethod
    def update(self, rng: np.random.Generator | torch.Generator):
        """Global update of the model. May call the update method of several transition kernels.

        Parameters
        ----------
        rng : np.random.Generator | torch.Generator
            Random number generator given by the sampler.
        """
        pass

    @abstractmethod
    def get_states(self) -> dict:
        """Extracts the current states of the variables in the model.

        Returns
        -------
        dict
            Current state of the variables.
        """
        pass

    @abstractmethod
    def set_states(self, states: dict):
        """Set the variables of the model the the values given in entry.

        Parameters
        ----------
        states : dict
            Dictionnary containing a new state for the variables of the model.
        """
        pass

    @abstractmethod
    def compute_potential(self) -> float:
        """Compute the potential of the targeted law.

        Returns
        -------
        float
            Potential.
        """
        pass

    @abstractmethod
    def aggregate_states(self):
        pass


class BaseDistributedModel(BaseModel):
    global_sizes: dict
    slices: dict

    @abstractmethod
    def set_slices(self):
        pass

    @abstractmethod
    def set_global_sizes(self):
        pass
