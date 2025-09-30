r"""Abstract GPU implementation for the Proximal Stochastic Gradient Langevin
Algorithm (PSGLA) :cite:p:`Salim2020`.
"""

# TODO refactor after gpu_context deletion

import cupy as cp
import torch
from typing import Optional, Any

from mcmc.transition_kernel.base_transition_kernel import BaseGpuTransitionKernel


class GpuPSGLA(BaseGpuTransitionKernel):
    """Generic GPU implementation of the Proximal Stochastic Gradient Langevin
    Algorithm (PSGLA) :cite:p:`Salim2020`.

    Attributes
    ----------
    step_size : float
        Step-size value used in the transition.

    Methods
    -------
    prox(cp.ndarray)
        Compute the proximity operator of the non-smooth term in the negative log-posterior function.
    grad(cp.ndarray)
        Compute the gradient of the differentiable term in the negative log-posterior function.
    """

    def __init__(
        self,
        dims,
        step_size,
        dtype: Optional[Any],
        initial_value: Optional[cp.ndarray] = None,
    ):
        r"""
        Parameters
        ----------
        dims : tuple[int, ...]
            Shape of the parameter handled by the transition kernel.
        step_size : float
            Step-size value used in the transition.
        dtype : cp.dtype | None, optional
            Parameter type, by default None.
        initial_value : cp.ndarray | None, optional
            Initial value for the chain (optional).
        """
        super().__init__(dims, initial_value, dtype=dtype)

        self.step_size = step_size

    # NOTE: The methods prox and grad should return at this stage, and be
    # defined by the user in any script where this class is actually usedd
    # https://stackoverflow.com/questions/10374527/dynamically-assigning-function-implementation-in-python

    def grad(self, state: cp.ndarray) -> cp.ndarray:
        """Compute the gradient of the differentiable term in the negative
        log-posterior function. Needs to be defined by the user."""
        raise ValueError("Warning : gradient function not defined!")

    def prox(self, state: cp.ndarray) -> cp.ndarray:
        """Compute the proximity operator of the non-smooth term in the
        negative log-posterior function. Needs to be defined by the user."""
        raise ValueError("Warning : proximal operator not defined!")

    def mc_step(self, rng: torch.Generator) -> None:
        self.current_state = self.prox(
            self.current_state
            + (2 * self.step_size) ** 0.5
            * cp.from_dlpack(
                torch.normal(
                    torch.zeros(self.current_state.shape, device="cuda"),
                    torch.ones(self.current_state.shape, device="cuda"),
                    generator=rng,
                )
            )
            - self.step_size * self.grad(self.current_state)
        )
