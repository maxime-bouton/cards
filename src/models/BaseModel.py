from abc import ABC, abstractmethod
from numpy import random

""" Abstract class that descibes the interface of the model class.
The methods declared here will be used by the sampler.
"""

class BaseModel(ABC):
    
    @abstractmethod
    def update(self, rng : random.Generator) -> None :
        """update Gobal update of the model. May call the update method of several transition kernel.

        Parameters
        ----------
        rng : random.Generator
            Random number generator given by the sampler.
        """
        return NotImplemented

    @abstractmethod
    def get_states(self) -> dict :
        """get_states Extracts the current states of the variables in the model.

        Returns
        -------
        dict
            Current state of the variables.
        """
        pass

    @abstractmethod
    def set_states(self, states : dict) -> None :
        """set_states Set the variables of the model the the values given in entry.

        Parameters
        ----------
        states : dict
            Dictionnary containing a new state for the variables of the model.
        """
        pass

    @abstractmethod
    def compute_potential(self) -> float :
        """compute_potential Compute the potential of the targeted law.

        Returns
        -------
        float
            Potential.
        """
        pass

    #! to be deleted, becomes useless when estimator is built inside the model.
    @abstractmethod
    def give_data2estimator(self):
        pass