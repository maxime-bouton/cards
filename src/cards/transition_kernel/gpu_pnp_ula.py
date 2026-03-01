r"""Abstract GPU implementation of the Plug-and-Play Unadjusted Langevin
(PnP-ULA) algorithm :cite:p:`Laumont2022`."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.
#
# adpated from: https://gitlab.cristal.univ-lille.fr/pthouven/dsgs


# TODO: documentation

import torch

from cards.backend import xp
from cards.transition_kernel.base_transition_kernel import BaseGpuTransitionKernel


class GpuPnpULA(BaseGpuTransitionKernel):
    def __init__(
        self,
        state_shape: tuple[int, ...],
        step_size: float,
        reg_coef: float,
        epsilon: float,
        lambda_: float,
        dtype: xp.dtype | None = None,
        initialization: xp.ndarray | None = None,
    ) -> None:
        super().__init__(state_shape, dtype=dtype, initial_value=initialization)
        self.step_size = step_size
        self.reg_coef = reg_coef
        self.lambda_ = lambda_
        self.epsilon = epsilon

    def denoise(self, state: xp.ndarray) -> xp.ndarray:
        raise ValueError("Warning : denoiser not defined!")

    def grad(self, state: xp.ndarray) -> xp.ndarray:
        raise ValueError("Warning : gradient function not defined!")

    def project(self, state: xp.ndarray) -> xp.ndarray:
        raise ValueError("Warning : projection function not defined!")

    def mc_step(self, rng):
        d = self.current_state - self.denoise(self.current_state)
        p = self.current_state - self.project(self.current_state)
        self.current_state = (
            self.current_state
            + (2 * self.step_size) ** 0.5
            * xp.from_dlpack(
                torch.normal(
                    mean=0,
                    std=1,
                    size=self.current_state.shape,
                    generator=rng,
                    device=rng.device,
                )
                # TODO: proper dtype handling in the torch.normal call
                # dtype=self.current_state.dtype,
            )
            - self.step_size
            * (
                self.grad(self.current_state)
                + d * self.reg_coef / self.epsilon
                + p / self.lambda_
            )
        )
