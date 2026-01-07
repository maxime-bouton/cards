r"""Abstract GPU implementation for the Proximal Stochastic Gradient Langevin
Algorithm (PSGLA) :cite:p:`Salim2020`.
"""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.
#
# adpated from: https://gitlab.cristal.univ-lille.fr/pthouven/dsgs

# NOTE: reference
# https://docs.python.org/3/library/typing.html
# https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html

from typing import Optional

import torch

from cards.backend import xp
from cards.transition_kernel.base_transition_kernel import BaseGpuTransitionKernel


class GpuPSGLA(BaseGpuTransitionKernel):
    r"""Generic CPU implementation of the Proximal Stochastic Gradient Langevin
    Algorithm (PSGLA) :cite:p:`Salim2020`, with target distribution of the form

    .. math::
            \pi(x) \propto \exp( -f(x) - g(x)),

    with :math:`f \in \Gamma_0 (\mathbb{R}^N)` an :math:`L_f`-smooth function
    and :math:`g: \mathbb{R}^N \mapsto (-\infty, +\infty]` with a known
    proximal operator.

    Attributes
    ----------
    step_size : float
        Step-size value used in the PSGLA transition.

    Methods
    -------
    prox(cards.backend.xp.ndarray)
        Compute the proximity operator of the non-smooth term in the negative log-posterior probability density function.
    grad(cards.backend.xp.ndarray)
        Compute the gradient of the differentiable term in the negative log-posterior probability density function.
    """

    def __init__(
        self,
        state_shape: tuple[int, ...],
        step_size: float,
        dtype: Optional[xp.dtype] = None,
        initial_value: Optional[xp.ndarray] = None,
    ) -> None:
        r"""Constructor of the PSGLA class.

        Parameters
        ----------
        state_shape : tuple[int, ...]
            Shape of the parameter handled by the transition kernel.
        step_size : float
            Step-size value used in the transition.
        dtype : cards.backend.xp.dtype | None, optional
            Parameter type, by default None.
        initial_value : cards.backend.xp.ndarray | None, optional
            Initial value for the chain (optional).
        """
        super().__init__(state_shape, initial_value, dtype=dtype)
        self.step_size = step_size

    # NOTE: The methods prox and grad should return at this stage, and be
    # defined by the user in any script where this class is actually used
    # https://stackoverflow.com/questions/10374527/dynamically-assigning-function-implementation-in-python

    def grad(self, state: xp.ndarray) -> xp.ndarray:
        """Compute the gradient of the differentiable term in the negative
        log-posterior function. Needs to be defined by the user."""
        raise ValueError("Warning : gradient function not defined!")

    def prox(self, state: xp.ndarray) -> xp.ndarray:
        """Compute the proximity operator of the non-smooth term in the
        negative log-posterior function. Needs to be defined by the user."""
        raise ValueError("Warning : proximal operator not defined!")

    def mc_step(self, rng: torch.Generator) -> None:
        self.current_state = self.prox(
            self.current_state
            + (2 * self.step_size) ** 0.5
            * xp.from_dlpack(
                torch.normal(
                    torch.zeros(self.current_state.shape, device="cuda"),
                    torch.ones(self.current_state.shape, device="cuda"),
                    generator=rng,
                )
                # TODO: proper dtype handling in the torch.normal call
                # https://docs.pytorch.org/docs/stable/generated/torch.set_default_dtype.html#torch.set_default_dtype
                # dtype=self.current_state.dtype,
            )
            - self.step_size * self.grad(self.current_state)
        )
