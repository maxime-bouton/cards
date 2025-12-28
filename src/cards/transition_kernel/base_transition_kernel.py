r"""Abstract class to implement probability transition kernels."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.

from abc import ABC, abstractmethod
from typing import Optional

from cards.backend import xp


class BaseTransitionKernel(ABC):
    r"""Abstract transition kernel class to support the development of generic
    MCMC algorithms.

    Attributes
    ----------
    current_state : cards.backend.xp.ndarray
        Current state of the parameter handled by the transition kernel.

    Methods
    -------
    mc_step()
        Update the state of the parameter.
    get_state()
        Return current state of the parameter.
    """

    def __init__(
        self,
        dims: tuple[int, ...],
        initial_value: Optional[xp.ndarray] = None,
        dtype: Optional[xp.dtype] = None,
    ) -> None:
        r"""
        Parameters
        ----------
        dims : Tuple[int, ...]
            Shape of the parameter handled by the transition kernel.
        initial_value : cp.ndarray | None, optional
            Initial value for the chain (optional).
        dtype : cards.backend.xp.dtype | None, optional
            Parameter type, by default None.
        """
        if initial_value is not None:
            self.current_state = initial_value
        else:
            if dtype is not None:
                self.current_state = xp.zeros(dims, dtype=dtype)
            else:
                self.current_state = xp.zeros(dims)

    @abstractmethod
    def mc_step(self, rng) -> None:
        r"""Update the state of the parameter."""
        pass

    def get_state(self) -> xp.ndarray | None:
        r"""Return current state of the parameter."""
        return self.current_state


class BaseGpuTransitionKernel(BaseTransitionKernel):
    def get_state(self) -> xp.ndarray | None:
        r"""Return current state of the parameter."""
        return self.current_state.get()
