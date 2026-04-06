r"""Abstract class to implement probability transition kernels."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
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
    dtype : cards.backend.xp.dtype
        Parameter type, by default None.

    Methods
    -------
    mc_step()
        Update the state of the parameter.
    get_state()
        Return current state of the parameter.
    """

    def __init__(
        self,
        state_shape: tuple[int, ...],
        initial_value: Optional[xp.ndarray] = None,
        dtype: Optional[xp.dtype] = None,
    ) -> None:
        r"""
        Parameters
        ----------
        state_shape : Tuple[int, ...]
            Shape of the parameter handled by the transition kernel.
        initial_value : cp.ndarray | None, optional
            Initial value for the chain (optional).
        dtype : cards.backend.xp.dtype | None, optional
            Parameter type, by default None.
        """
        self.dtype = dtype
        if initial_value is not None:
            self.current_state = initial_value.astype(self.dtype)
        else:
            self.current_state = xp.zeros(state_shape, dtype=self.dtype)

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
