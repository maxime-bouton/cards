"""Abstact class to implement transition kernels for MCMC algorithms."""

from abc import ABC, abstractmethod

from mcmc.backend import gpu_context, xp


class BaseTransitionKernel(ABC):
    def __init__(self, dims, gpu_id=0):
        with gpu_context(gpu_id):
            self.current_state = xp.zeros(dims)

    @abstractmethod
    def mc_step(self, rng) -> None:
        pass

    def get_state(self):
        return self.current_state


class BaseGpuTransitionKernel(BaseTransitionKernel):
    def get_state(self):
        return self.current_state.get()
