"""Implementation of the pseudo-gradient Langevin algorithm."""

from mcmc.backend import xp
from mcmc.transition_kernel.base_transition_kernel import BaseTransitionKernel


class PSGLA(BaseTransitionKernel):
    def __init__(
        self,
        state_shape: tuple[int, ...],
        step_size: float,
        dtype: xp.dtype | None = None,
    ):
        super(PSGLA, self).__init__(state_shape, dtype=dtype)
        self.step_size = step_size
        # FIXME: add prox parameter here, so that it can be taken into account directly in mc_step, and not rewritten each time in the implementation of prox (prox_step = step_size * prox_parameter)
        # FIXME: add default method to compute step-size from Lipshitz constant?

    # NOTE: The methods prox and grad should return at this stage, and be
    # defined by the user in any script where this class is actually usedd
    # https://stackoverflow.com/questions/10374527/dynamically-assigning-function-implementation-in-python

    def prox(self, state: xp.ndarray) -> xp.ndarray:
        raise ValueError("Warning : proximal operator not defined!")

    def grad(self, state: xp.ndarray) -> xp.ndarray:
        raise ValueError("Warning : gradient function not defined!")

    def mc_step(self, rng):
        self.current_state = self.prox(
            self.current_state
            + (2 * self.step_size) ** 0.5
            * rng.standard_normal(self.current_state.shape)
            - self.step_size * self.grad(self.current_state)
        )
