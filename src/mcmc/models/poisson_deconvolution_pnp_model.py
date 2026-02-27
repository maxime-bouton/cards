import numpy as np
from mpi4py import MPI

from mcmc.backend import xp
from mcmc.denoisers.base_denoiser import BaseDenoiser, BaseDistributedDenoiser
from mcmc.functionals.prox import KL, prox_KL, prox_nonegativity
from mcmc.models.base_model import BaseDistributedModel
from mcmc.models.base_poisson_deconvolution_model import (
    BasePoissonDeconvolutionModel,
    PoissonDeconvolutionParameters,
)
from mcmc.operators.dft_convolution import DftConvolution
from mcmc.operators.mpi_dft_convolution import MpiDftConvolution
from mcmc.transition_kernel.base_transition_kernel import (
    BaseTransitionKernel,
)
from mcmc.transition_kernel.gpu_pnp_sgla import GpuPnpSGLA
from mcmc.transition_kernel.gpu_pnp_ula import GpuPnpULA
from mcmc.transition_kernel.gpu_psgla import GpuPSGLA
from mcmc.transition_kernel.psgla import PSGLA


class BasePoissonDeconvolutionPnpModel(BasePoissonDeconvolutionModel):
    def __init__(
        self,
        params: PoissonDeconvolutionParameters,
        X: BaseTransitionKernel,
        Z1: BaseTransitionKernel,
        Z2: BaseTransitionKernel,
        denoiser: BaseDenoiser,
    ):
        self.denoiser = denoiser
        super().__init__(params, X, Z1, Z2)

    def set_conditionals(self) -> None:
        """Set the conditionals of the transition kernels including the coupling between those kernels."""
        if (type(self.X) is PSGLA) or (type(self.X) is GpuPSGLA):
            self.X.prox = prox_nonegativity
            self.X.grad = (
                lambda state: self.dynamic_range**2
                * self.convolution_operator.adjoint(
                    self.convX - self.Z1.current_state / self.dynamic_range
                )
                / self.split_coef1
                + (state - self.Z2.current_state) / self.split_coef2
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

        if (type(self.Z1) is PSGLA) or (type(self.Z1) is GpuPSGLA):
            self.Z1.prox = lambda state: (
                prox_KL(state, self.observations, lam=self.Z1.step_size)
            )
            self.Z1.grad = (
                lambda state: (state - self.dynamic_range * self.convX)
                / self.split_coef1
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

        if type(self.Z2) is GpuPnpULA:
            self.Z2.denoise = lambda state: self.denoiser(state, self.Z2.epsilon**0.5)
            self.Z2.grad = (
                lambda state: (state - self.X.current_state) / self.split_coef2
            )
            self.Z2.project = lambda state: state.clip(-1, 2)
        elif type(self.Z2) is GpuPnpSGLA:
            self.Z2.denoise = lambda state: self.denoiser(
                state, self.Z2.reg_coef * self.Z2.epsilon**0.5
            )
            self.Z2.grad = (
                lambda state: (state - self.X.current_state) / self.split_coef2
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
        return {
            "X": self.X.get_state(),
            "Z1": self.Z1.get_state(),
            "Z2": self.Z2.get_state(),
            **self._get_estimator_builder_states(),
        }

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
        self.Z1.current_state = xp.asarray(states["Z1"])
        self.Z2.current_state = xp.asarray(states["Z2"])
        self.convX = self.convolution_operator.forward(self.X.current_state)

    def update(self, rng: np.random.Generator):
        """update Gobal update of the model. Updates every kernel used by the model and computes annex variables.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator, given by the sampler.
        """
        self.X.mc_step(rng)
        self.convX = self.convolution_operator.forward(self.X.current_state)
        self.Z1.mc_step(rng)
        self.Z2.mc_step(rng)

    def compute_potential(self) -> float:
        """compute_potential Computes the potential.

        Returns
        -------
        float
            Potential of the targeted law.
        """
        p = KL(self.Z1.current_state, self.observations)
        p += xp.sum(self.Z1.current_state - self.dynamic_range * self.convX) ** 2 / (
            2 * self.split_coef1
        )
        p += xp.sum((self.X.current_state - self.Z2.current_state) ** 2) / (
            2 * self.split_coef2
        )
        return p


class PoissonDeconvolutionPnpModel(BasePoissonDeconvolutionPnpModel):
    def __init__(
        self,
        params: PoissonDeconvolutionParameters,
        X: BaseTransitionKernel,
        Z1: BaseTransitionKernel,
        Z2: BaseTransitionKernel,
        denoiser: BaseDenoiser,
    ):
        self.convolution_operator = DftConvolution(
            np.asarray(X.current_state.shape), params.kernel, params.observations.shape
        )

        super().__init__(params, X, Z1, Z2, denoiser)


class DistributedPoissonDeconvolutionPnpModel(
    BasePoissonDeconvolutionPnpModel,
    BaseDistributedModel,
):
    def __init__(
        self,
        comm: MPI.Comm,
        full_size: np.ndarray,
        grid_size: np.ndarray,
        params: PoissonDeconvolutionParameters,
        X: BaseTransitionKernel,
        Z1: BaseTransitionKernel,
        Z2: BaseTransitionKernel,
        denoiser: BaseDistributedDenoiser,
    ):
        self.comm = comm
        self.full_size = full_size

        self.convolution_operator = MpiDftConvolution(
            self.full_size,
            params.kernel,
            self.comm,
            grid_size,
            tile_range=denoiser.tile_range,
        )
        super().__init__(params, X, Z1, Z2, denoiser)

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
        slices["Z1"] = (
            self.convolution_operator.adjoint_communicator.cartslicer._get_slice_global_buffer_to_tile()
        )
        slices["Z2"] = self.denoiser.global_to_tile_slice

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
        sizes["Z1"] = np.asarray(
            self.convolution_operator.adjoint_communicator.cartslicer.global_buffer_size,
            dtype=int,
        )
        sizes["Z2"] = np.asarray(self.full_size, dtype=int)

        for key in self.estimator_builder.get_keys():
            var = key.split("_")[0]
            if key.endswith("_samples"):
                sizes[key] = np.r_[self.estimator_builder._batch_size, sizes[var]]
            else:
                sizes[key] = sizes[var]

        self.global_sizes = sizes
