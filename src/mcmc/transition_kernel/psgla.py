r"""Abstract implementation for the Proximal Stochastic Gradient Langevin
Algorithm (PSGLA) :cite:p:`Salim2020`.
"""

from mcmc.backend import xp
from mcmc.transition_kernel.base_transition_kernel import BaseTransitionKernel
from typing import Optional, Any

# TODO: fuse with gpu_psgla (only differs through one instruction)


class PSGLA(BaseTransitionKernel):
    """Generic implementation of the Proximal Stochastic Gradient Langevin
    Algorithm (PSGLA) :cite:p:`Salim2020`.

    Attributes
    ----------
    step_size : float
        Step-size value used in the transition.

    Methods
    -------
    prox(xp.ndarray)
        Compute the proximity operator of the non-smooth term in the negative log-posterior function.
    grad(xp.ndarray)
        Compute the gradient of the differentiable term in the negative log-posterior function.
    """

    def __init__(self, dims, step_size, dtype: Optional[Any] = None):
        r"""
        Parameters
        ----------
        dims : tuple[int, ...]
            Shape of the parameter handled by the transition kernel.
        step_size : float
            Step-size value used in the transition.
        dtype : xp.dtype | None, optional
            Parameter type, by default None.
        """
        super(PSGLA, self).__init__(dims, dtype=dtype)
        self.step_size = step_size
        # FIXME: add prox parameter here, so that it can be taken into account directly in mc_step, and not rewritten each time in the implementation of prox (prox_step = step_size * prox_parameter)
        # FIXME: add default method to compute step-size from Lipshitz constant?

    # NOTE: The methods prox and grad should return at this stage, and be
    # defined by the user in any script where this class is actually usedd
    # https://stackoverflow.com/questions/10374527/dynamically-assigning-function-implementation-in-python

    def prox(self, state: xp.ndarray) -> xp.ndarray:
        """Compute the proximity operator of the non-smooth term in the
        negative log-posterior function. Needs to be defined by the user."""
        raise ValueError("Warning : proximal operator not defined!")

    def grad(self, state: xp.ndarray) -> xp.ndarray:
        """Compute the gradient of the differentiable term in the negative
        log-posterior function. Needs to be defined by the user."""
        raise ValueError("Warning : gradient function not defined!")

    def mc_step(self, rng):
        self.current_state = self.prox(
            self.current_state
            + (2 * self.step_size) ** 0.5
            * rng.standard_normal(self.current_state.shape)
            - self.step_size * self.grad(self.current_state)
        )
