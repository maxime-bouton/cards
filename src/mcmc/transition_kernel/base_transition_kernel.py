"""Abstract class to implement transition kernels for MCMC algorithms."""

from abc import ABC, abstractmethod

from mcmc.backend import xp


class BaseTransitionKernel(ABC):
    def __init__(
        self,
        state_shape: tuple[int, ...],
        dtype: xp.dtype | None = None,
        initialization: xp.ndarray | None = None,
    ):
        self.current_state = xp.zeros(state_shape, dtype=dtype)

        if initialization is not None:
            self.current_state[:] = initialization

    @abstractmethod
    def mc_step(self, rng) -> None:
        pass

    def get_state(self):
        return self.current_state


class BaseGpuTransitionKernel(BaseTransitionKernel):
    def get_state(self):
        return self.current_state.get()
