r"""Abstract CPU implementation for the Proximal Stochastic Gradient Langevin
Algorithm (PSGLA) :cite:p:`Salim2020`.
"""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.
#
# adpated from: https://gitlab.cristal.univ-lille.fr/pthouven/dsgs

# TODO: documentation
# TODO: fuse with gpu_psgla (only differs through one instruction)

from typing import Optional

from cards.backend import xp
from cards.transition_kernel.base_transition_kernel import BaseTransitionKernel


class PSGLA(BaseTransitionKernel):
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
    prox(xp.ndarray)
        Compute the proximity operator of the non-smooth term in the negative log-posterior probability density function.
    grad(xp.ndarray)
        Compute the gradient of the differentiable term in the negative log-posterior probability density function.
    """

    def __init__(
        self,
        state_shape: tuple[int, ...],
        step_size: float,
        dtype: Optional[xp.dtype] = None,
        initial_value: xp.ndarray | None = None,
    ) -> None:
        r"""Constructor of the PSGLA class.

        Parameters
        ----------
        state_shape : tuple[int, ...]
            Shape of the parameter handled by the transition kernel.
        step_size : float
            Step-size value used in the transition.
        dtype : xp.dtype | None, optional
            Parameter type, by default None.
        initial_value: xp.ndarray | None, optional
            Initial parameter value, by default None.
        """
        super(PSGLA, self).__init__(
            state_shape, dtype=dtype, initial_value=initial_value
        )
        self.step_size = step_size
        # FIXME: add prox parameter here, so that it can be taken into account directly in mc_step, and not rewritten each time in the implementation of prox (prox_step = step_size * prox_parameter)
        # FIXME: add default method to compute step-size from Lipschitz constant?

    # NOTE: The methods prox and grad should return at this stage, and be
    # defined by the user in any script where this class is actually used
    # https://stackoverflow.com/questions/10374527/dynamically-assigning-function-implementation-in-python

    def prox(self, state: xp.ndarray) -> xp.ndarray:
        """Compute the proximity operator of the non-smooth term in the
        negative log-posterior function. Needs to be defined by the user."""
        raise ValueError("Warning : proximal operator not defined!")

    def grad(self, state: xp.ndarray) -> xp.ndarray:
        """Compute the gradient of the differentiable term in the negative
        log-posterior function. Needs to be defined by the user."""
        raise ValueError("Warning : gradient function not defined!")

    def mc_step(self, rng) -> None:
        self.current_state = self.prox(
            self.current_state
            + (2 * self.step_size) ** 0.5
            * rng.standard_normal(self.current_state.shape, dtype=self.dtype)
            - self.step_size * self.grad(self.current_state)
        )
