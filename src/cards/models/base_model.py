r"""Define abstract base classes for models and associated sampling strategies.

This module provides the common interface for all models, specifying how an application
interacts with its underlying transition kernels for a specific inference problem. The
methods declared here are intended to be called directly within a
:class:`~cards.samplers.base_sampler.BaseSampler` instance.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch

import cards.backend as xp


class BaseModel(ABC):
    r"""Abstract base class for inference models.

    This class defines the common interface for models, managing the associated
    estimators and specifying the required methods for state manipulation, potential
    computation, and transition kernel updates.
    """

    @abstractmethod
    def update(self, rng: np.random.Generator | torch.Generator) -> None:
        r"""Update the current state of all random variables involved in the model.

        This method triggers the transition mechanism (i.e., the update method) of each
        underlying transition kernel.

        Parameters
        ----------
        rng : np.random.Generator | torch.Generator
            Random number generator provided by the sampler.
        """

    @abstractmethod
    def get_states(self) -> dict[str, xp.ndarray]:
        r"""Extract the current states of all variables being sampled from the model.

        Returns
        -------
        dict[str, xp.ndarray]
            The current states of the variables.
        """

    @abstractmethod
    def set_states(self, states: dict[str, xp.ndarray]) -> None:
        r"""Set the model variables to the associated values provided in the input.

        Parameters
        ----------
        states : dict[str, xp.ndarray]
            Dictionary containing the new state values for the model's variables.
        """

    @abstractmethod
    def compute_potential(self) -> float:
        r"""Compute the potential function :math:`-\log p(\mathbf{x} \mid \mathbf{y})`
        for the target posterior distribution.

        Returns
        -------
        float
            The current value of the potential function.

        Note
        ----
        The potential function cannot be evaluated for target distributions associated
        with a learned prior (e.g., encoded by a deep denoiser in Plug-and-Play approaches).
        """

    @property
    def vars(self) -> list[str]:
        r"""Return the list of variables to be sampled in the model."""
        return list(self.get_states().keys())


class BaseDistributedModel(BaseModel):
    r"""Abstract base class for distributed inference models.

    This class extends :class:`BaseModel` to support distributed sampling across
    multiple MPI workers. It manages the allocation and slicing of global
    and local memory buffers.

    .. warning::
        Inheriting classes **must** call :meth:`super().__init__` at the
        very end of their own :meth:`__init__` method.

        The parent constructor automatically calls the abstract setup methods
        (:meth:`set_slices`, :meth:`set_global_sizes`, and :meth:`set_local_sizes`).
        Because these methods typically rely on attributes defined in the child class,
        calling the parent constructor too early will result in missing attribute errors.
    """

    def __init__(self):
        super().__init__()
        self.global_sizes = {}
        self.local_sizes = {}
        self.slices = {}

        self.set_slices()
        self.set_global_sizes()
        self.set_local_sizes()

    @abstractmethod
    def set_slices(self) -> None:
        r"""Set the slicer to select local variables from the global memory buffer."""

    @abstractmethod
    def set_global_sizes(self) -> None:
        r"""Set the sizes of the global buffers representing the variables to be sampled."""

    @abstractmethod
    def set_local_sizes(self) -> None:
        r"""Set the sizes of the local buffers representing the portion of the variables
        handled by each MPI worker."""
