r"""Implementation of a Gaussian inpainting model using a PnP prior to reproduce the experiments reported in :cite:p:`Bouton2025`."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.

# TODO: documentation

from dataclasses import dataclass

import numpy as np
import torch
from mpi4py import MPI

from cards.backend import xp
from cards.denoisers.base_denoiser import BaseDenoiser
from cards.models.base_gaussian_inpainting_model import (
    BaseGaussianInpaintingModel,
    GaussianInpaintingParameters,
)
from cards.models.base_model import BaseDistributedModel
from cards.transition_kernel.base_transition_kernel import (
    BaseTransitionKernel,
)
from cards.transition_kernel.gpu_pnp_sgla import GpuPnpSGLA
from cards.transition_kernel.gpu_pnp_ula import GpuPnpULA


@dataclass
class GaussianInpaintingPnpParameters(GaussianInpaintingParameters): ...


class BaseGaussianInpaintingPnpModel(BaseGaussianInpaintingModel):
    def __init__(
        self,
        params: GaussianInpaintingPnpParameters,
        X: BaseTransitionKernel,
        denoiser: BaseDenoiser,
    ):
        self.denoiser = denoiser
        super().__init__(params, X)

    def set_conditionals(self):
        """Set the conditionals of the transition kernels including the coupling between those kernels."""
        if type(self.X) is GpuPnpULA:
            self.X.denoise = lambda state: self.denoiser(
                state,
                self.X.epsilon**0.5,
                torch_dtype=torch.float32,
                cp_dtype=xp.float64,
            )
            self.X.grad = lambda state: (
                self.mask * (state - self.observations) / self.sigma2
            )
            self.X.project = lambda state: state.clip(-1, 2)
        elif type(self.X) is GpuPnpSGLA:
            self.X.denoise = lambda state: self.denoiser(
                state,
                self.X.reg_coef * self.X.epsilon**0.5,
                torch_dtype=torch.float32,
                cp_dtype=xp.float64,
            )
            self.X.grad = lambda state: (
                self.mask * (state - self.observations) / self.sigma2
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

    def get_states(self) -> dict:
        """Extracts the current state of the transition kernel and other variables of interest and return the in a dictionary.

        Returns
        -------
        dict
            Dictionary containing the curent states of the variables.
        """
        return {"X": self.X.get_state(), "MMSE": self.estimator_builder.estimator.get()}

    def set_states(self, states: dict) -> None:
        """set_states
        Read the dictionary given in entry and set the variables of the model to the values contained in it.
        The keys used by the dictionary must be the same as in "get_states"

        Parameters
        ----------
        states : dict
            Dictionary containing new values for the variables of the model.
        """
        self.X.current_state = xp.asarray(states["X"])

    def update(self, rng: np.random.Generator):
        """update Gobal update of the model. Updates every kernel used by the model and computes annex variables.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator, given by the sampler.
        """
        self.X.mc_step(rng)

    def compute_potential(self) -> float:
        """compute_potential Computes the potential.

        Returns
        -------
        float
            Potential of the targeted law.
        """
        p = xp.sum((self.observations - self.mask * self.X.current_state) ** 2) / (
            2 * self.sigma2
        )
        return p


class GaussianInpaintingPnpModel(BaseGaussianInpaintingPnpModel): ...


class DistributedGaussianInpaintingPnpModel(
    BaseGaussianInpaintingPnpModel,
    BaseDistributedModel,
):
    def __init__(
        self,
        comm: MPI.Comm,
        full_size: np.ndarray,
        params: GaussianInpaintingPnpParameters,
        X: BaseTransitionKernel,
        denoiser: BaseDenoiser,
    ):
        self.comm = comm
        self.full_size = full_size

        super().__init__(params, X, denoiser)

    def set_slices(self):
        """set_slices Describes which portion of the global buffer the current thread must handle.

        Returns
        -------
        dict
            Dictionary containing the slices of the global buffer that this thread will handle.
        """
        self.slices["X"] = self.denoiser.global_to_tile_slice
        self.slices["MMSE"] = self.denoiser.global_to_tile_slice

    def set_global_sizes(self):
        """set_global_sizes Describe the gobla sizes of several global buffers.

        Returns
        -------
        dict
            Global sizes of the variable of interest.
        """
        self.global_sizes["X"] = np.asarray(self.full_size, dtype=int)
        self.global_sizes["MMSE"] = np.asarray(self.full_size, dtype=int)

    def set_local_sizes(self):
        self.local_sizes["X"] = self.X.current_state.shape
        self.local_sizes["MMSE"] = self.X.current_state.shape
