r"""Implementation of a Poisson deconvolution model using a TV prior to reproduce the experiments reported in :cite:p:`Bouton2026`."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

# TODO: documentation

import numpy as np

import cards.backend as xp
from cards.estimators.base_estimator import BaseEstimator
from cards.functionals.prox import (
    KL,
    l21_norm,
    prox_KL,
    prox_l21norm,
    prox_nonegativity,
)
from cards.models.base_model import BaseDistributedModel
from cards.models.base_poisson_deconvolution_model import (
    BasePoissonDeconvolutionModel,
    PoissonDeconvolutionParameters,
)
from cards.operators.dft_convolution import DftConvolution
from cards.operators.distributed_gradient import DistributedGradient2d
from cards.operators.gradient import Gradient2d
from cards.operators.mpi_dft_convolution import MpiDftConvolution
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel
from cards.transition_kernels.gpu_psgla import GpuPSGLA
from cards.transition_kernels.psgla import PSGLA


class BasePoissonDeconvolutionTvModel(BasePoissonDeconvolutionModel):
    def __init__(
        self,
        estimators: list[BaseEstimator],
        params: PoissonDeconvolutionParameters,
        convolution_operator: DftConvolution | MpiDftConvolution,
        gradient_operator: Gradient2d | DistributedGradient2d,
        X: BaseTransitionKernel,
        Z1: BaseTransitionKernel,
        Z2: BaseTransitionKernel,
    ):
        self.gradient_operator = gradient_operator

        self.gradX = xp.zeros_like(X.current_state)
        super().__init__(estimators, params, convolution_operator, X, Z1, Z2)

    def set_conditionals(self) -> None:
        """Set the conditionals of the transition kernels including the coupling between those kernels."""
        if (type(self.X) is PSGLA) or (type(self.X) is GpuPSGLA):
            self.X.prox = prox_nonegativity
            self.X.grad = lambda state: (
                self.dynamic_range**2
                * self.convolution_operator.adjoint(
                    self.convX - self.Z1.current_state / self.dynamic_range
                )
                / self.split_coef1
                + self.gradient_operator.adjoint(self.gradX - self.Z2.current_state)
                / self.split_coef2
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

        if (type(self.Z1) is PSGLA) or (type(self.Z1) is GpuPSGLA):
            self.Z1.prox = lambda state: prox_KL(
                state, self.observations, lam=self.Z1.step_size
            )
            self.Z1.grad = lambda state: (
                (state - self.dynamic_range * self.convX) / self.split_coef1
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

        if (type(self.Z2) is PSGLA) or (type(self.Z2) is GpuPSGLA):
            self.Z2.prox = lambda state: prox_l21norm(
                state, lam=self.Z2.step_size * self.reg_coeff
            )
            self.Z2.grad = lambda state: (state - self.gradX) / self.split_coef2
        else:
            raise ValueError("Kernel type not yet supported by this model.")

    def get_states(self) -> dict:
        """Extracts the current state of the transition kernel and other variables of interest and return the in a dictionary.

        Returns
        -------
        dict
            Dictionary containing the curent states of the variables.
        """
        return {
            "X": self.X.get_state(),
            "Z1": self.Z1.get_state(),
            "Z2": self.Z2.get_state(),
        }

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
        self.Z1.current_state = xp.asarray(states["Z1"])
        self.Z2.current_state = xp.asarray(states["Z2"])

        self.gradX = self.gradient_operator.forward(self.X.current_state)
        self.convX = self.convolution_operator.forward(self.X.current_state)

    # TODO: revise typing here
    def update(self, rng: np.random.Generator):
        """update Gobal update of the model. Updates every kernel used by the model and computes annex variables.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator, given by the sampler.
        """
        self.X.mc_step(rng)
        self.gradX = self.gradient_operator.forward(self.X.current_state)
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
        p += xp.sum((self.gradX - self.Z2.current_state) ** 2) / (2 * self.split_coef2)
        p += self.reg_coeff * l21_norm(self.Z2.current_state)
        return p


class PoissonDeconvolutionTvModel(BasePoissonDeconvolutionTvModel):
    def __init__(
        self,
        estimators: list[BaseEstimator],
        params: PoissonDeconvolutionParameters,
        convolution_operator: DftConvolution,
        X: BaseTransitionKernel,
        Z1: BaseTransitionKernel,
        Z2: BaseTransitionKernel,
    ):
        gradient_operator = Gradient2d(np.array([*X.current_state.shape]))

        super().__init__(
            estimators,
            params,
            convolution_operator,
            gradient_operator,
            X,
            Z1,
            Z2,
        )


class DistributedPoissonDeconvolutionTvModel(
    BasePoissonDeconvolutionTvModel,
    BaseDistributedModel,
):
    def __init__(
        self,
        estimators: list[BaseEstimator],
        params: PoissonDeconvolutionParameters,
        convolution_operator: MpiDftConvolution,
        X: BaseTransitionKernel,
        Z1: BaseTransitionKernel,
        Z2: BaseTransitionKernel,
    ):
        self.full_size = convolution_operator.image_size

        gradient_operator = DistributedGradient2d(
            convolution_operator.image_size,
            convolution_operator.grid_size,
            convolution_operator.comm,
        )

        super().__init__(
            estimators,
            params,
            convolution_operator,
            gradient_operator,
            X,
            Z1,
            Z2,
        )

    def set_slices(self):
        """set_slices Describes which portion of the global buffer the current thread must handle.

        Returns
        -------
        dict
            Dictionary containing the slices of the global buffer that this thread will handle.
        """
        self.slices["X"] = (
            self.gradient_operator.cart_comm.cartslicer.slice_global_buffer_to_tile
        )
        self.slices["Z1"] = (
            self.convolution_operator.adjoint_communicator.cartslicer.slice_global_buffer_to_tile
        )
        self.slices["Z2"] = (
            np.s_[:],
            *self.gradient_operator.cart_comm.cartslicer.slice_global_buffer_to_tile,
        )

    def set_global_sizes(self):
        """set_global_sizes Describe the gobla sizes of several global buffers.

        Returns
        -------
        dict
            Global sizes of the variable of interest.
        """
        self.global_sizes["X"] = np.asarray(self.full_size, dtype=int)
        self.global_sizes["Z1"] = np.asarray(
            self.convolution_operator.adjoint_communicator.cartslicer.global_buffer_size,
            dtype=int,
        )
        self.global_sizes["Z2"] = np.asarray([2, *self.full_size], dtype=int)

    def set_local_sizes(self):
        self.local_sizes["X"] = self.X.current_state.shape
        self.local_sizes["Z1"] = self.Z1.current_state.shape
        self.local_sizes["Z2"] = self.Z2.current_state.shape
