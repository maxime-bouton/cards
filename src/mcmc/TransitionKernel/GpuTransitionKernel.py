import cupy as cp
import torch
import numpy as np

from abc import ABC, abstractmethod


class BaseGpuTransitionKernel(ABC):
    def __init__(self, dims):
        self.current_state = cp.zeros(dims)

    @abstractmethod
    def mc_step(self, rng) -> None:
        return NotImplemented


# ! any way to avoid copy/paste?
class GpuPSGLA(BaseGpuTransitionKernel):
    def __init__(self, dims, step_size):
        super(GpuPSGLA, self).__init__(dims)
        self.step_size = step_size

    def prox(self, state: cp.ndarray) -> cp.ndarray:
        print("Warning : proximal operator not defined !")
        return NotImplemented

    def grad(self, state: cp.ndarray) -> cp.ndarray:
        print("Warning : gradient function not defined!")
        return NotImplemented

    def mc_step(self, rng):
        self.current_state = self.prox(
            self.current_state
            + cp.sqrt(2 * self.step_size)
            * cp.from_dlpack(
                torch.normal(
                    torch.zeros(self.current_state.shape, device="cuda"),
                    torch.ones(self.current_state.shape, device="cuda"),
                    generator=rng,
                )
            )
            - self.step_size * self.grad(self.current_state)
        )


class MultiGpuPSGLA(GpuPSGLA):
    def __init__(self, dims, step_size, gpu_id):
        super().__init__(dims, step_size)
        self.gpu_id = gpu_id
        with cp.cuda.Device(self.gpu_id):
            self.current_state = cp.zeros(dims)

    def device_prox(self, state: cp.ndarray, gpu_id: int) -> cp.ndarray:
        with cp.cuda.Device(gpu_id):
            return self.prox(state)

    def device_grad(self, state: cp.ndarray, gpu_id: int):
        with cp.cuda.Device(gpu_id):
            return self.grad(state)

    def grad(self, state: cp.ndarray) -> cp.ndarray:
        print("Warning : gradient function not defined!")
        return NotImplemented

    def mc_step(self, rng):
        with cp.cuda.Device(self.gpu_id):
            self.current_state = self.device_prox(
                self.current_state
                + ( self.step_size **(0.5) )
                * cp.from_dlpack(
                    torch.normal(
                        torch.zeros(self.current_state.shape, device="cuda"),
                        torch.ones(self.current_state.shape, device="cuda"),
                        generator=rng,
                    )
                )
                - self.step_size * self.device_grad(self.current_state, self.gpu_id),
                self.gpu_id,
            )
