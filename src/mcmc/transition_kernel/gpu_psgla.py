r"""Abstract GPU implementation for the Proximal Stochastic Gradient Langevin
Algorithm (PSGLA) :cite:p:`Salim2020`.
"""

import cupy as cp
import torch

from mcmc.backend import gpu_context
from mcmc.transition_kernel.base_transition_kernel import BaseGpuTransitionKernel


class GpuPSGLA(BaseGpuTransitionKernel):
    def __init__(self, dims, step_size, gpu_id=-1):
        super().__init__(dims, step_size)

        self.step_size = step_size
        self.gpu_id = gpu_id

        with gpu_context(self.gpu_id):
            self.current_state = cp.zeros(dims)

    # NOTE: The methods prox and grad should return at this stage, and be
    # defined by the user in any script where this class is actually usedd
    # https://stackoverflow.com/questions/10374527/dynamically-assigning-function-implementation-in-python

    def device_prox(self, state: cp.ndarray) -> cp.ndarray:
        with gpu_context(self.gpu_id):
            return self.prox(state)

    def device_grad(self, state: cp.ndarray):
        with gpu_context(self.gpu_id):
            return self.grad(state)

    def grad(self, state: cp.ndarray) -> cp.ndarray:
        raise ValueError("Warning : gradient function not defined!")

    def prox(self, state: cp.ndarray) -> cp.ndarray:
        raise ValueError("Warning : proximal operator not defined!")

    def mc_step(self, rng: torch.Generator) -> None:
        with gpu_context(self.gpu_id):
            self.current_state = self.device_prox(
                self.current_state
                + (2 * self.step_size) ** 0.5
                * cp.from_dlpack(
                    torch.normal(
                        torch.zeros(self.current_state.shape, device="cuda"),
                        torch.ones(self.current_state.shape, device="cuda"),
                        generator=rng,
                    )
                )
                - self.step_size * self.device_grad(self.current_state)
            )
