from abc import ABC, abstractmethod

import numpy as np


class BaseSerialTransitionKernel(ABC):
    def __init__(self, dims):
        self.current_state = np.zeros(dims)

    @abstractmethod
    def mc_step(self, rng):
        pass


class PSGLA(BaseSerialTransitionKernel):
    def __init__(self, dims, step_size):
        super(PSGLA, self).__init__(dims)
        self.step_size = step_size
        # FIXME: add prox parameter here, so that it can be taken into account directly in mc_step, and not rewritten each time in the implementation of prox (prox_step = step_size * prox_parameter)
        # FIXME: add default method to compute step-size from Lipshitz constant?

    # FIXME: return an error, not just print! The method should be astract at this stage
    # https://stackoverflow.com/questions/10374527/dynamically-assigning-function-implementation-in-python
    def prox(self, state: np.ndarray) -> np.ndarray:
        print("Warning : proximal operator not defined !")
        return NotImplemented

    def grad(self, state: np.ndarray) -> np.ndarray:
        print("Warning : gradient function not defined!")
        return NotImplemented

    def mc_step(self, rng):
        self.current_state = self.prox(
            self.current_state
            + np.sqrt(2 * self.step_size)
            * rng.standard_normal(self.current_state.shape)
            - self.step_size * self.grad(self.current_state)
        )


class MYULA(BaseSerialTransitionKernel):
    def __init__(self, dims, step_size, reg_coeff):
        super(PSGLA, self).__init__(dims)
        self.step_size = step_size
        self.reg_coeff = reg_coeff
        # FIXME: add default method to compute step-size from Lipshitz constant?

    def prox(self, state: np.ndarray) -> np.ndarray:
        print("Warning : proximal operator has not be defined !")
        return NotImplemented

    def grad(self, state: np.ndarray) -> np.ndarray:
        print("Warning : gradient function has not be defined !")
        return NotImplemented

    def mc_step(self, rng):
        self.current_state = (
            self.current_state
            + np.sqrt(2 * self.step_size)
            * rng.standard_normal(self.current_state.shape)
            - self.step_size
            * (
                self.grad(self.current_state)
                + self.prox(self.current_state) / self.reg_coeff
            )
        )
