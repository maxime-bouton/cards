"""Implementation of the Moreau-Yoshida Unajusted Langevin Algorithm."""

import numpy as np

from mcmc.backend import xp
from mcmc.transition_kernel.base_transition_kernel import BaseTransitionKernel


class MYULA(BaseTransitionKernel):
    def __init__(self, dims, step_size, reg_coeff, dtype: np.dtype | None = None):
        super(MYULA, self).__init__(dims, dtype=dtype)
        self.step_size = step_size
        self.reg_coeff = reg_coeff
        # FIXME: add default method to compute step-size from Lipshitz constant?

    def prox(self, state: xp.ndarray) -> xp.ndarray:
        raise ValueError("Warning : proximal operator has not be defined !")

    def grad(self, state: xp.ndarray) -> xp.ndarray:
        raise ValueError("Warning : gradient function has not be defined !")

    def mc_step(self, rng):
        self.current_state = (
            self.current_state
            + (2 * self.step_size) ** 0.5
            * rng.standard_normal(self.current_state.shape)
            - self.step_size
            * (
                self.grad(self.current_state)
                + self.prox(self.current_state) / self.reg_coeff
            )
        )
