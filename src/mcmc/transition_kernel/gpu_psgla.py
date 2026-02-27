"""Implementation of the pseudo-gradient Langevin algorithm."""

import cupy as cp
import torch

from mcmc.transition_kernel.base_transition_kernel import BaseGpuTransitionKernel


class GpuPSGLA(BaseGpuTransitionKernel):
    def __init__(
        self,
        state_shape: tuple[int, ...],
        step_size: float,
        dtype: cp.dtype | None = None,
        initialization: cp.ndarray | None = None,
    ):
        super().__init__(state_shape, dtype=dtype, initialization=initialization)

        self.step_size = step_size

    # NOTE: The methods prox and grad should return at this stage, and be
    # defined by the user in any script where this class is actually usedd
    # https://stackoverflow.com/questions/10374527/dynamically-assigning-function-implementation-in-python

    def device_prox(self, state: cp.ndarray) -> cp.ndarray:
        return self.prox(state)

    def device_grad(self, state: cp.ndarray):
        return self.grad(state)

    def grad(self, state: cp.ndarray) -> cp.ndarray:
        raise ValueError("Warning : gradient function not defined!")

    def prox(self, state: cp.ndarray) -> cp.ndarray:
        raise ValueError("Warning : proximal operator not defined!")

    def mc_step(self, rng: torch.Generator) -> None:
        self.current_state = self.device_prox(
            self.current_state
            + (2 * self.step_size) ** 0.5
            * cp.asarray(
                torch.normal(
                    mean=0,
                    std=1,
                    size=self.current_state.shape,
                    generator=rng,
                    device=rng.device,
                )
            )
            - self.step_size * self.device_grad(self.current_state)
        )
