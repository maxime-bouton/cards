r"""Abstract class that descibes the interface of the model class. The methods
declared here will be called within the sampler.
"""

from abc import ABC, abstractmethod

from numpy import random


class BaseModel(ABC):
    @abstractmethod
    def update(self, rng: random.Generator):
        """update Global update of the model. May call the update method of several transition kernels.

        Parameters
        ----------
        rng : random.Generator
            Random number generator given by the sampler.
        """
        pass

    @abstractmethod
    def get_states(self) -> dict:
        """get_states Extracts the current states of the variables in the model.

        Returns
        -------
        dict
            Current state of the variables.
        """
        pass

    def get_states4batch(self) -> dict:
        """get_states4batch Return a dictionnary containing the current state of the variables. It will be used to store and save all the states of those vairiables along the chain.

        Returns
        -------
        dict
            Dictionnary containing the current states of variables.
        """
        return {}

    def get_batch_sizes(self) -> dict:
        """get_batch_sizes Only called when we need to save the full batch. Return a dictionnary containing the dimensions of the variables we want to save.

        Returns
        -------
        dict
            Dictionnary continaing the dimensions of the variables we want to save.
        """
        return {}

    @abstractmethod
    def set_states(self, states: dict):
        """set_states Set the variables of the model the the values given in entry.

        Parameters
        ----------
        states : dict
            Dictionnary containing a new state for the variables of the model.
        """
        return NotImplemented

    @abstractmethod
    def compute_potential(self) -> float:
        """compute_potential Compute the potential of the targeted law.

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
    global_sizes = dict
    slices = dict

    @abstractmethod
    def set_slices(self):
        """set_slices Describes the local selection of local variables in regards to the global variables."""
        pass

    @abstractmethod
    def set_global_sizes(self):
        """set_global_sizes Describes the sizes of global variables."""
        pass

    @abstractmethod
    def set_local_sizes(self):
        """set_local_sizes Describes the sizes of the local variables."""
        pass
