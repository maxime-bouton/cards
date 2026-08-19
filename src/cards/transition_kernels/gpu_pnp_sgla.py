r"""Abstract GPU implementation of the Plug-and-Playp proximal stochastic
gradient Langevin algorithm (PnP-PSGLA) :cite:p:`Renaud2025`.
"""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

import torch

import cards.backend as xp
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel


class GpuPnpSGLA(BaseTransitionKernel):
    r"""PnP-PSGLA transition kernel.

    PnP-PSGLA transition kernel :cite:p:`Renaud2025`, whose negative
    log-likelihood is assumed to be an :math:`L_f`-smooth function.

    Attributes
    ----------
    step_size : float
        Step-size value used in the PnP-PSGLA transition.
    """

    def __init__(
        self,
        state_shape: tuple[int, ...],
        step_size: float,
        dtype: xp.dtype | None = None,
        initial_value: xp.ndarray | None = None,
    ) -> None:
        r"""Constructor of the GpuPnpSGLA class.

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
        super().__init__(state_shape, dtype=dtype, initial_value=initial_value)
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

    def mc_step(self, rng: torch.Generator):
        self.current_state = self.denoise(
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
                # https://docs.pytorch.org/docs/stable/generated/torch.set_default_dtype.html#torch.set_default_dtype
                # https://github.com/pytorch/pytorch/issues/40568
            )
            - self.step_size * self.grad(self.current_state)
        )
