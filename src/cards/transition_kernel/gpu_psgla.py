r"""Abstract GPU implementation of the Proximal Stochastic Gradient Langevin
Algorithm (PSGLA) :cite:p:`Salim2020`.
"""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.

from typing import Optional

import torch

from cards.backend import xp
from cards.transition_kernel.base_transition_kernel import BaseGpuTransitionKernel


class GpuPSGLA(BaseGpuTransitionKernel):
    r"""Generic GPU implementation of the Proximal Stochastic Gradient Langevin
    Algorithm (PSGLA) :cite:p:`Salim2020`, with target distribution

    .. math::
            \pi(x) \propto \exp( -f(x) - g(x)),

    with :math:`f \in \Gamma_0 (\mathbb{R}^N)` an :math:`L_f`-smooth function
    and :math:`g: \mathbb{R}^N \mapsto (-\infty, +\infty]` with a known
    proximal operator.

    Attributes
    ----------
    step_size : float
        Step-size value used in the PSGLA transition.
    """

    def __init__(
        self,
        state_shape: tuple[int, ...],
        step_size: float,
        dtype: Optional[xp.dtype] = None,
        initial_value: Optional[xp.ndarray] = None,
    ) -> None:
        r"""Constructor of the GpuPSGLA class.

        Parameters
        ----------
        state_shape : tuple[int, ...]
            Shape of the parameter handled by the transition kernel.
        step_size : float
            Step-size value used in the transition.
        dtype : cards.backend.xp.dtype | None, optional
            Parameter type, by default None.
        initial_value : cards.backend.xp.ndarray | None, optional
            Initial state value, by default None.
        """
        super().__init__(state_shape, dtype=dtype, initial_value=initial_value)
        self.step_size = step_size

    # NOTE: prox and grad should be defined by the user
    # https://stackoverflow.com/questions/10374527/dynamically-assigning-function-implementation-in-python

    def prox(self, state: xp.ndarray) -> xp.ndarray:
        r"""Compute the proximity operator of the non-smooth term in the
        negative log-posterior function. To be defined by the user.
        """
        raise ValueError("Proximal operator not defined.")

    def grad(self, state: xp.ndarray) -> xp.ndarray:
        r"""Compute the gradient of the differentiable term in the negative
        log-posterior function. To be defined by the user.
        """
        raise ValueError("Gradient function not defined.")

    def mc_step(self, rng: torch.Generator) -> None:
        self.current_state = self.prox(
            self.current_state
            + (2 * self.step_size) ** 0.5
            * xp.from_dlpack(
                torch.normal(
                    mean=0,
                    std=1,
                    size=self.current_state.shape,
                    generator=rng,
                    device=rng.device,
                )
                # TODO: proper dtype handling in the torch.normal call
                # https://docs.pytorch.org/docs/stable/generated/torch.set_default_dtype.html#torch.set_default_dtype
                # dtype=self.current_state.dtype,
            )
            - self.step_size * self.grad(self.current_state)
        )
