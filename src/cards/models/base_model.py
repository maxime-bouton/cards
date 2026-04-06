r"""Abstract class specifying the interface common to all models, used to
define an application and the sampling strategy selected. Methods in the class
encode the interaction between the transition kernels underlying a sampler tailored to a specific inference problem. The methods declared here are called
within the sampler.
"""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def update(self, rng):
        r"""Updates all the current state of all the random variables involved in the model. This method triggers the update method (i.e., transition mechanism) of each underlying transition kernels.

        Parameters
        ----------
        rng : cards.backend.xp.random.Generator | torch.Generator
            Random number generator given by a sampler.
        """
        pass

    @abstractmethod
    def get_states(self) -> dict:
        r"""Extracts the current state of all the variables to be sampled from the model.

        Returns
        -------
        dict
            Current state of the variables.
        """
        pass

    @abstractmethod
    def get_states4batch(self) -> dict:
        r"""Returns a dictionary containing a batch of samples for all the variables to be sampled from the model.

        The dictionary stores a batch of consecutive samples for all the variables to be sampled, to be saved to disk.

        Returns
        -------
        dict
            Dictionary containing a batch of consecutive samples for each variable.
        """
        pass

    @abstractmethod
    def get_batch_sizes(self) -> dict:
        r"""Returns a dictionary containing the dimensions of the variables to be saved to disk.

        This method is only called when a batch of samples needs to be saved to disk.

        Returns
        -------
        dict
            Dictionary continaing the dimensions of the variables we want to save.
        """
        pass

    @abstractmethod
    def set_states(self, states: dict):
        r"""Set the variables in the model to the associated value passed in input.

        Parameters
        ----------
        states : dict
            Dictionary containing a new state value for the variables of the model.
        """
        pass

    @abstractmethod
    def compute_potential(self) -> float:
        r"""Compute the potential function :math:`-\log p(\mathbf{x} \mid \mathbf{y})` for the target posterior distribution.

        Returns
        -------
        float
            Current value of the potential function.

        Note
        ----
        The potential function cannot be evaluated for target distributions
        associated with a learned prior, e.g., encoded by a deep denoiser in PnP approaches.
        """
        pass

    @abstractmethod
    def aggregate_states(self):
        r"""Aggregate consecutive samples over a predefined window to progressively form parameter estimates (e.g., MMSE estimate).

        Note
        ----
        The method triggers the :meth:`~cards.estimator.BaseEstimatorBuilder.aggregate_states` in the estimators selected for the application of interest.
        """
        pass


class BaseDistributedModel(BaseModel):
    # FIXME: see if declaration as class variables really needed
    global_sizes: dict
    slices: dict

    @abstractmethod
    def set_slices(self):
        r"""Sets the slicer to select local variables from the global memory buffer."""
        pass

    @abstractmethod
    def set_global_sizes(self):
        r"""Sets the sizes of global buffers representing the variables to be sampled."""
        pass

    @abstractmethod
    def set_local_sizes(self):
        r"""Sets the sizes of the local buffers representing the portion of the variables handled by each MPI worker."""
        pass
