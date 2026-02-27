from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from mcmc.backend import xp
from mcmc.denoisers.base_denoiser import BaseDenoiser
from mcmc.models.base_gaussian_inpainting_model import (
    BaseGaussianInpaintingModel,
    GaussianInpaintingParameters,
)
from mcmc.models.base_model import BaseDistributedModel
from mcmc.transition_kernel.base_transition_kernel import (
    BaseTransitionKernel,
)
from mcmc.transition_kernel.gpu_pnp_sgla import GpuPnpSGLA
from mcmc.transition_kernel.gpu_pnp_ula import GpuPnpULA


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
            self.X.denoise = lambda state: self.denoiser(state, self.X.epsilon**0.5)
            self.X.grad = (
                lambda state: self.mask * (state - self.observations) / self.sigma2
            )
            self.X.project = lambda state: state.clip(-1, 2)
        elif type(self.X) is GpuPnpSGLA:
            self.X.denoise = lambda state: self.denoiser(
                state, self.X.reg_coef * self.X.epsilon**0.5
            )
            self.X.grad = (
                lambda state: self.mask * (state - self.observations) / self.sigma2
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

    def get_states(self) -> dict:
        """Extracts the current state of the transition kernel and other variables of interest and return the in a dictionnary.

        Returns
        -------
        dict
            Dictionnary containing the curent states of the variables.
        """
        return {"X": self.X.get_state(), **self._get_estimator_builder_states()}

    def set_states(self, states: dict) -> None:
        """set_states
        Read the dictionnary given in entry and set the variables of the model to the values contained in it.
        The keys used by the dictionnary must be the same as in "get_states"

        Parameters
        ----------
        states : dict
            Dictionnary containing new values for the variables of the model.
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

        self.slices = {}
        self.global_sizes = {}

    def set_slices(self):
        """set_slices Describes which portion of the global buffer the current thread must handle.

        Returns
        -------
        dict
            Dictionary containing the slices of the global buffer that this thread will handle.
        """
        slices = {}
        slices["X"] = self.denoiser.global_to_tile_slice

        for key in self.estimator_builder.get_keys():
            var = key.split("_")[0]
            if key.endswith("_samples"):
                slices[key] = (np.s_[:],) + slices[var]
            else:
                slices[key] = slices[var]

        self.slices = slices

    def set_global_sizes(self):
        """set_global_sizes Describe the gobla sizes of several global buffers.

        Returns
        -------
        dict
            Global sizes of the variable of interest.
        """
        sizes = {}
        sizes["X"] = np.asarray(self.full_size, dtype=int)

        for key in self.estimator_builder.get_keys():
            var = key.split("_")[0]
            if key.endswith("_samples"):
                sizes[key] = np.r_[self.estimator_builder._batch_size, sizes[var]]
            else:
                sizes[key] = sizes[var]

        self.global_sizes = sizes
