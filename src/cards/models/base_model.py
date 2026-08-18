r"""Define abstract base classes for models and associated sampling strategies.

This module provides the common interface for all models, specifying how an application
interacts with its underlying transition kernels for a specific inference problem. The
methods declared here are intended to be called directly within a
:class:`~cards.samplers.base_sampler.BaseSampler` instance.
"""

from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
import torch

import cards.backend as xp
from cards.core.variable import Variable


class BaseModel(ABC):
    r"""Abstract base class for inference models.

    This class defines the common interface for models, managing the associated
    estimators and specifying the required methods for state manipulation, potential
    computation, and transition kernel updates.
    """

    def __init__(self, *variables: Variable):
        super().__init__()
        self._variables = {}

        for var in variables:
            if var.name in self._variables:
                raise ValueError(
                    f"Duplicate variable name detected: '{var.name}'. "
                    "All variables passed to the model must have unique names."
                )
            self._variables[var.name] = var

    @property
    def keys(self):
        r"""Return a list containing the names of all variables being sampled."""
        return list(self._variables.keys())

    @property
    def states(self) -> dict[str, xp.ndarray]:
        r"""Extract the current states of all variables being sampled from the model.

        Returns
        -------
        dict[str, xp.ndarray]
            The current states of the variables.
        """
        return {var.name: var.state for var in self._variables.values()}

    @states.setter
    def states(self, states: dict[str, xp.ndarray]) -> None:
        r"""Set the model variables to the associated values provided in the input.

        Parameters
        ----------
        states : dict[str, xp.ndarray]
            Dictionary containing the new state values for the model's variables.
        """
        for k, v in states.items():
            self._variables[k].state[:] = v

        self._on_states_updated()

    def _on_states_updated(self) -> None:
        r"""Update all cached buffers in the model's variables."""

    @abstractmethod
    def compile(self) -> None:
        r"""Compile the model to prepare it for sampling."""

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


class BaseDistributedModel(BaseModel):
    r"""Base class for distributed inference models.

    This class extends :class:`BaseModel` to support distributed sampling across
    multiple MPI workers. It manages the allocation and slicing of global
    and local memory buffers.
    """

    @cached_property
    def local_sizes(self) -> dict:
        r"""Return sizes of the local buffers representing the portion of the variables
        handled by each MPI worker."""
        return {n: v.layout.tile for n, v in self._variables.items()}

    @cached_property
    def global_sizes(self) -> dict:
        r"""Return the sizes of the global buffers representing the variables to be sampled."""
        return {n: v.layout.full for n, v in self._variables.items()}

    @cached_property
    def slices(self) -> dict:
        r"""Return the slices to select local variables from the global memory buffer."""
        return {n: v.layout.s for n, v in self._variables.items()}
