r"""Abstract CPU implementation for the Proximal Stochastic Gradient Langevin
Algorithm (PSGLA) :cite:p:`Salim2020`.
"""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)
#
# adpated from: https://gitlab.cristal.univ-lille.fr/pthouven/dsgs

# TODO: fuse with GpuPSGLA if possible

from typing import Optional

import cards.backend as xp
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel


class PSGLA(BaseTransitionKernel):
    r"""Abstract CPU implementation of the Proximal Stochastic Gradient Langevin
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
        r"""PSGLA constructor.

        Parameters
        ----------
        state_shape : tuple[int, ...]
            Shape of the parameter handled by the transition kernel.
        step_size : float
            Step-size value used in the transition.
        dtype : cards.backend.xp.dtype | None, optional
            Parameter type, by default None.
        initial_value: cards.backend.xp.ndarray | None, optional
            Initial state value, by default None.
        """
        super(PSGLA, self).__init__(
            state_shape, dtype=dtype, initial_value=initial_value
        )
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

    def mc_step(self, rng: xp.random.Generator) -> None:
        self.current_state = self.prox(
            self.current_state
            + (2 * self.step_size) ** 0.5
            * rng.standard_normal(self.current_state.shape, dtype=self.dtype)
            - self.step_size * self.grad(self.current_state)
        )
