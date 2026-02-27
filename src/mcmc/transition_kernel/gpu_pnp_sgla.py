"""Implementation of the Plug-and-Play Unadjusted Langevin algorithm."""

import cupy as cp
import torch

from mcmc.transition_kernel.base_transition_kernel import BaseGpuTransitionKernel


class GpuPnpSGLA(BaseGpuTransitionKernel):
    def __init__(
        self,
        state_shape: tuple[int, ...],
        step_size: float,
        reg_coef: float,
        epsilon: float,
        dtype: cp.dtype | None = None,
    ):
        super().__init__(state_shape, dtype=dtype)
        self.step_size = step_size
        self.reg_coef = reg_coef
        self.epsilon = epsilon

    def denoise(self, state: cp.ndarray) -> cp.ndarray:
        raise ValueError("Warning : denoiser not defined!")

    def grad(self, state: cp.ndarray) -> cp.ndarray:
        raise ValueError("Warning : gradient function not defined!")

    def mc_step(self, rng):
        self.current_state = self.denoise(
            self.current_state
            + (2 * self.step_size) ** 0.5
            * cp.asarray(
                torch.normal(
                    mean=0,
                    std=1,
                    size=self.current_state.shape,
                    generator=rng,
                    device=rng.device,
                ),
                # TODO: proper dtype handling in the torch.normal call
                dtype=self.current_state.dtype,
            )
            - self.step_size * self.grad(self.current_state)
        )
