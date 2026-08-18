r"""Abstract class to implement probability transition kernels."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.

from abc import ABC, abstractmethod

import numpy as np
import torch

import cards.backend as xp
from cards.core.variable import Variable


class BaseTransitionKernel(ABC):
    r"""Abstract transition kernel class to support the development of generic
    MCMC algorithms.

    Parameters
    ----------
    var : ~cards.core.variable.Variable
        Variable to be sampled by the transition kernel.

    Methods
    -------
    mc_step()
        Update the state of the parameter.
    """

    def __init__(self, var: Variable) -> None:
        self.var = var

    @abstractmethod
    def mc_step(self, rng: np.random.Generator | torch.Generator) -> None:
        r"""Update the state of the parameter."""

    @property
    def state(self) -> xp.ndarray:
        r"""Return current state of the parameter."""
        return self.var.state
