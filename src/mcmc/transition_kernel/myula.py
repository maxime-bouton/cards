r"""Abstract implementation for the Moreau-Yoshida Unajusted Langevin Algorithm (PSGLA) :cite:p:`Durmus2018`."""

from mcmc.backend import xp
from mcmc.transition_kernel.base_transition_kernel import BaseTransitionKernel


class MYULA(BaseTransitionKernel):
    def __init__(
        self,
        dims,
        lipschitz_cst: float,
        reg_factor: float = 1.0,
        step_factor: float = 0.5,
    ):
        super(MYULA, self).__init__(dims)

        assert reg_factor <= 1
        assert step_factor <= 1

        self.lipschitz_cst = lipschitz_cst

        self.reg_cst = reg_factor / lipschitz_cst
        self.step_size = step_factor / (lipschitz_cst + 1 / self.reg_cst)

        self.cst_1 = (2 * self.step_size) ** 0.5
        self.cst_2 = self.step_size / self.reg_cst

    def set_params(self, new_reg_factor: float = 1, new_step_factor: float = 0.5):
        assert new_reg_factor <= 1
        assert new_step_factor <= 1

        self.reg_cst = new_reg_factor / self.lipschitz_cst
        self.step_size = new_step_factor / (self.lipschitz_cst + 1 / self.reg_cst)

        self.cst_1 = (2 * self.step_size) ** 0.5
        self.cst_2 = self.step_size / self.reg_cst

    def set_lipschitz(
        self, new_lipschitz: float, reg_factor: float = 1, step_factor: float = 0.5
    ):
        self.lipschitz_cst = new_lipschitz

        self.set_params(reg_factor, step_factor)

    def prox(self, state: xp.ndarray) -> xp.ndarray:
        """Compute the proximity operator of the non-smooth term in the
        negative log-posterior function. Needs to be defined by the user."""
        raise ValueError("Warning : proximal operator has not be defined !")

    def grad(self, state: xp.ndarray) -> xp.ndarray:
        """Compute the gradient of the differentiable term in the negative
        log-posterior function. Needs to be defined by the user."""
        raise ValueError("Warning : gradient function has not be defined !")

    def mc_step(self, rng):
        self.current_state = (
            (1 - self.cst_2) * self.current_state
            - self.step_size * self.grad(self.current_state)
            + self.cst_2 * self.prox(self.current_state)
            + self.cst_1 * rng.standard_normal(self.current_state.shape)
        )
