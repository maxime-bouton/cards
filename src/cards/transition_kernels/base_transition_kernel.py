r"""Abstract class to implement probability transition kernels."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from abc import ABC, abstractmethod

import cards.backend as xp


class BaseTransitionKernel(ABC):
    r"""Abstract transition kernel class to support the development of generic
    MCMC algorithms.

    Attributes
    ----------
    current_state : cards.backend.xp.ndarray
        Current state of the parameter handled by the transition kernel.
    dtype : cards.backend.xp.dtype
        Numeric type for the state, by default None (default configuration).

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
        initial_value: xp.ndarray | None = None,
        dtype: xp.dtype | None = None,
    ) -> None:
        r"""
        Parameters
        ----------
        state_shape : Tuple[int, ...]
            Shape of the parameter handled by the transition kernel.
        initial_value : xp.ndarray | None, optional
            Initial state value, by default None.
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

    def get_state(self) -> xp.ndarray | None:
        r"""Return current state of the parameter."""
        return self.current_state
