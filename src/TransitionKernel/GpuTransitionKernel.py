import numpy as np
import cupy as cp
import torch

from abc import ABC, abstractmethod



class BaseGpuTransitionKernel(ABC):
    def __init__(self, dims):
        self.current_state = cp.zeros( dims )

    @abstractmethod
    def mc_step(self, rng) -> None:
        return NotImplemented

# ! any way to avoid copy/paste?
class GpuPSGLA( BaseGpuTransitionKernel ):
    def __init__(self, dims, step_size):
        super(GpuPSGLA, self).__init__(dims)
        self.step_size = step_size

    def prox(self, state : cp.ndarray) ->  cp.ndarray :
        print("Warning : proximal operator not defined !")
        return NotImplemented
    
    def grad(self, state : cp.ndarray) ->  cp.ndarray :
        print("Warning : gradient function not defined!")
        return NotImplemented

    def mc_step(self, rng):
        #self.current_state = self.prox(  self.current_state + np.sqrt(2*self.step_size)*rng.standard_normal( self.current_state.shape ) \
        #                               - self.step_size*self.grad( self.current_state) )
        self.current_state = self.prox(  self.current_state \
                                        + np.sqrt(2*self.step_size)\
                                        *cp.from_dlpack( torch.normal( torch.zeros(self.current_state.shape, device='cuda'), torch.ones(self.current_state.shape, device='cuda'),generator=rng) )\
                                       - self.step_size*self.grad( self.current_state) )