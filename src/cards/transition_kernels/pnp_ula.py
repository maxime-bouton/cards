r"""Implementations of the Plug-and-Play Unadjusted Langevin
(PnP-ULA) algorithm :cite:p:`Laumont2022`.
"""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from abc import abstractmethod

import numpy as np
import torch

import cards.backend as xp
from cards.core.variable import Variable
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel


class PnpULA(BaseTransitionKernel):
    r"""Generic implementation of PnP-ULA.

    Parameters
    ----------
    step_size : float
        Step-size value used in the transition.
    reg_coef : float
        Regularization parameter for the contribution from Tweedie's
        identity (MMSE denoiser).
    epsilon : float
        Standard deviation of the denoiser.
    lambda_ : float
        Projection smoothing parameter.

    Attributes
    ----------
    step_size : float
        Step-size value used in the transition.
    reg_coef : float
        Regularization parameter for the contribution from Tweedie's
        identity (MMSE denoiser).
    epsilon : float
        Standard deviation of the denoiser.
    lambda_ : float
        Projection smoothing parameter.
    """

    def __init__(
        self,
        var: Variable,
        step_size: float,
        reg_coef: float,
        epsilon: float,
        lambda_: float,
    ) -> None:
        r"""Constructor of the GpuPnpULA class."""
        super().__init__(var)
        self.step_size = step_size
        self.reg_coef = reg_coef
        self.lambda_ = lambda_
        self.epsilon = epsilon

    def denoise(self, state: xp.ndarray) -> xp.ndarray:
        r"""Apply denoiser specified by the user."""
        raise ValueError("Denoiser not defined.")

    def grad(self, state: xp.ndarray) -> xp.ndarray:
        r"""Compute the gradient of the differentiable term in the negative
        log-posterior function. To be defined by the user.
        """
        raise ValueError("Gradient function not defined.")

    def project(self, state: xp.ndarray) -> xp.ndarray:
        r"""Project samples onto a predefined compact set. To be defined by the
        user.
        """
        raise ValueError("Projection function not defined.")

    @abstractmethod
    def _noise(
        self,
        state: xp.ndarray,
        rng,
    ) -> xp.ndarray: ...

    def mc_step(self, rng):
        state = self.var.state

        grad = self.grad(state)
        denoised = self.denoise(state)
        projected = self.project(state)
        noise = self._noise(state, rng)

        xp.subtract(state, denoised, out=denoised)
        denoised *= self.reg_coef / self.epsilon

        xp.subtract(state, projected, out=projected)
        projected /= self.lambda_

        grad += denoised
        grad += projected
        grad *= self.step_size

        state -= grad

        noise *= (2 * self.step_size) ** 0.5
        state += noise


class CpuPnpULA(PnpULA):
    def _noise(self, state: xp.ndarray, rng: np.random.Generator) -> xp.ndarray:
        return rng.standard_normal(state.shape, dtype=state.dtype)


class GpuPnpULA(PnpULA):
    def _noise(self, state: xp.ndarray, rng: torch.Generator) -> xp.ndarray:
        return xp.asarray(
            torch.normal(
                mean=0.0,
                std=1.0,
                size=state.shape,
                generator=rng,
                device=rng.device,
            ),
            state.dtype,
        )
