r"""Abstract CPU implementation for the Proximal Stochastic Gradient Langevin
Algorithm (PSGLA) :cite:p:`Salim2020`.
"""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)
#
# adpated from: https://gitlab.cristal.univ-lille.fr/pthouven/dsgs

from abc import abstractmethod

import numpy as np
import torch

import cards.backend as xp
from cards.core.variable import Variable
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel

# TODO: update documentation


class PSGLA(BaseTransitionKernel):
    r"""Abstract CPU implementation of the Proximal Stochastic Gradient Langevin
    Algorithm (PSGLA) :cite:p:`Salim2020`, with target distribution

    .. math::
            \pi(x) \propto \exp( -f(x) - g(x)),

    with :math:`f \in \Gamma_0 (\mathbb{R}^N)` an :math:`L_f`-smooth function
    and :math:`g: \mathbb{R}^N \mapsto (-\infty, +\infty]` with a known
    proximal operator.

    Parameters
    ----------
    var : Variable
        Shape of the parameter handled by the transition kernel.
    step_size : float
        Step-size value used in the transition.

    Attributes
    ----------
    var : Variable
        Shape of the parameter handled by the transition kernel.
    step_size : float
        Step-size value used in the PSGLA transition.
    """

    def __init__(
        self,
        var: Variable,
        step_size: float,
    ) -> None:
        super().__init__(var)
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

    @abstractmethod
    def _noise(
        self,
        state: xp.ndarray,
        rng,
    ) -> xp.ndarray: ...

    def mc_step(self, rng):
        state = self.var.state

        grad = self.grad(state)
        noise = self._noise(state, rng)

        self.var.state = self.prox(
            state - self.step_size * grad + (2 * self.step_size) ** 0.5 * noise
        )


class CpuPSGLA(PSGLA):
    def _noise(self, state: xp.ndarray, rng: np.random.Generator) -> xp.ndarray:
        return rng.standard_normal(state.shape, dtype=state.dtype)


class GpuPSGLA(PSGLA):
    def _noise(self, state: xp.ndarray, rng: torch.Generator) -> xp.ndarray:
        return xp.asarray(
            torch.normal(
                mean=0.0,
                std=1.0,
                size=state.shape,
                generator=rng,
                device=rng.device,
            ),
            state.dtype,
        )
