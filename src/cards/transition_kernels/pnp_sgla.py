r"""Implementation of the Plug-and-Playp proximal stochastic
gradient Langevin algorithm (PnP-PSGLA) :cite:p:`Renaud2025`.
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

# TODO: update documentation


class PnpSGLA(BaseTransitionKernel):
    r"""PnP-PSGLA transition kernel.

    PnP-PSGLA transition kernel :cite:p:`Renaud2025`, whose negative
    log-likelihood is assumed to be an :math:`L_f`-smooth function.

    Attributes
    ----------
    var : Variable
        ...
    step_size : float
        Step-size value used in the PnP-PSGLA transition.
    epsilon : float
        Standard deviation of the denoiser.

    Attributes
    ----------
    step_size : float
        Step-size value used in the PnP-PSGLA transition.
    epsilon : float
        Standard deviation of the denoiser.
    """

    def __init__(
        self,
        var: Variable,
        step_size: float,
    ) -> None:
        r"""Constructor of the PnpSGLA class.

        Parameters
        ----------
        state_shape : tuple[int, ...]
            Shape of the parameter handled by the transition kernel.
        step_size : float
            Step-size value used in the transition.
        dtype : xp.dtype | None, optional
            Parameter type, by default None.
        initial_value : xp.ndarray | None, optional
            Initial state value, by default None.
        """
        super().__init__(var)
        self.step_size = step_size

    def denoise(self, state: xp.ndarray) -> xp.ndarray:
        r"""Apply pre-trained denoiser involved in PnP-PSGLA.

        Parameters
        ----------
        state: cards.backend.xp.ndarray
            Input state.

        Returns
        -------
        cards.backend.xp.ndarray
            Denoiser output.

        Note
        ----
        To be defined by the user.
        """
        raise ValueError("Denoiser not defined!")

    def grad(self, state: xp.ndarray) -> xp.ndarray:
        r"""Compute the gradient of the differentiable term in the negative
        log-posterior probability density function.

        Parameters
        ----------
        state: cards.backend.xp.ndarray
            Input state.

        Returns
        -------
        cards.backend.xp.ndarray
            Evaluation of the gradient of the differentiable part in the neg-log poserior probability density function.

        Note
        ----
        To be defined by the user.
        """
        raise ValueError("Gradient function not defined.")

    @abstractmethod
    def _noise(
        self,
        state: xp.ndarray,
        rng,
    ) -> xp.ndarray: ...

    def mc_step(self, rng):
        state = self.var.state
        grad = self.grad(state)
        noise = self._noise(state, rng)
        self.var.state = self.denoise(
            state - self.step_size * grad + (2 * self.step_size) ** 0.5 * noise
        )


class CpuPnpSGLA(PnpSGLA):
    def _noise(self, state: xp.ndarray, rng: np.random.Generator) -> xp.ndarray:
        return rng.standard_normal(state.shape, dtype=state.dtype)


class GpuPnpSGLA(PnpSGLA):
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
