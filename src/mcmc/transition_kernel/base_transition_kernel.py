"""Abstact class to implement transition kernels for MCMC algorithms."""

from abc import ABC, abstractmethod

from mcmc.backend import xp
from typing import Optional, Any


class BaseTransitionKernel(ABC):
    r"""Abstract transition kernel class to support the development of MCMC
    algorithms.

    Attributes
    ----------
    current_state : xp.ndarray
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
        dims,
        initial_value: Optional[xp.ndarray] = None,
        dtype: Optional[Any] = None,
    ):
        r"""
        Parameters
        ----------
        dims : tuple[int, ...]
            Shape of the parameter handled by the transition kernel.
        initial_value : cp.ndarray | None, optional
            Initial value for the chain (optional).
        dtype : xp.dtype | None, optional
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
        """Update the state of the parameter."""
        pass

    def get_state(self):
        """Return current state of the parameter."""
        return self.current_state


class BaseGpuTransitionKernel(BaseTransitionKernel):
    def get_state(self):
        """Return current state of the parameter."""
        return self.current_state.get()
