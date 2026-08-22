r"""Implementation of a Gaussian inpainting model using a TV prior to reproduce the experiments reported in :cite:p:`Bouton2026`."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

# TODO: documentation
# FIXME: replace class variables by instance variables
# FIXME: update name of variables (grid_size -> mpi_grid_shape, unify interfaces w.r.t. array shapes)

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

import cards.backend as xp
from cards.functionals.prox import l21_norm, prox_l21norm, prox_nonegativity
from cards.models.base_gaussian_inpainting_model import (
    BaseGaussianInpaintingModel,
    GaussianInpaintingParameters,
)
from cards.models.base_model import BaseDistributedModel
from cards.operators.distributed_gradient import DistributedGradient2d
from cards.operators.gradient import Gradient2d
from cards.transition_kernels.base_transition_kernel import BaseTransitionKernel
from cards.transition_kernels.gpu_psgla import GpuPSGLA
from cards.transition_kernels.psgla import PSGLA


@dataclass
class GaussianInpaintingTvParameters(GaussianInpaintingParameters):
    split_coeff: float


class BaseGaussianInpaintingTvModel(BaseGaussianInpaintingModel):
    gradient_operator: Gradient2d | DistributedGradient2d

    def __init__(
        self,
        params: GaussianInpaintingTvParameters,
        X: BaseTransitionKernel,
        Z: BaseTransitionKernel,
    ):
        self.Z = Z
        self.split_coeff = params.split_coeff
        self.gradX = xp.zeros_like(X.current_state)
        super().__init__(params, X)

    def set_conditionals(self) -> None:
        """Set the conditionals of the transition kernels including the coupling between those kernels."""
        if (type(self.X) is PSGLA) or (type(self.X) is GpuPSGLA):
            self.X.prox = prox_nonegativity
            self.X.grad = lambda state: (
                self.mask * (state - self.observations) / self.sigma2
                + self.gradient_operator.adjoint(self.gradX - self.Z.current_state)
                / self.split_coeff
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

        if (type(self.Z) is PSGLA) or (type(self.Z) is GpuPSGLA):
            self.Z.prox = lambda state: prox_l21norm(
                state, lam=self.Z.step_size * self.reg_coeff, axis=0
            )
            self.Z.grad = lambda state: (state - self.gradX) / self.split_coeff
        else:
            raise ValueError("Kernel type not yet supported by this model.")

    def get_states(self) -> dict:
        """Extracts the current state of the transition kernel and other variables of interest and return the in a dictionary.

        Returns
        -------
        dict
            Dictionary containing the curent states of the variables.
        """
        return {"X": self.X.get_state(), "Z": self.Z.get_state()}

    def set_states(self, states: dict) -> None:
        """set_states
        Read the dictionary given in entry and set the variables of the model to the values contained in it.
        The keys used by the dictionary must be the same as in "get_states"

        Parameters
        ----------
        states : dict
            Dictionary containing new values for the variables of the model.
        """
        self.X.current_state = xp.asarray(states["X"], dtype=self.X.current_state.dtype)
        self.Z.current_state = xp.asarray(states["Z"], dtype=self.Z.current_state.dtype)
        self.gradX = self.gradient_operator.forward(self.X.current_state)

    def update(self, rng: np.random.Generator):
        """update Gobal update of the model. Updates every kernel used by the model and computes annex variables.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator, given by the sampler.
        """
        self.X.mc_step(rng)
        self.gradX = self.gradient_operator.forward(self.X.current_state)
        self.Z.mc_step(rng)

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
        p += xp.sum((self.gradX - self.Z.current_state) ** 2) / (2 * self.split_coeff)
        p += self.reg_coeff * l21_norm(self.Z.current_state)
        return p


class GaussianInpaintingTvModel(BaseGaussianInpaintingTvModel):
    def __init__(
        self,
        params: GaussianInpaintingTvParameters,
        X: BaseTransitionKernel,
        Z: BaseTransitionKernel,
    ):
        self.gradient_operator = Gradient2d(np.array([*X.current_state.shape]))

        super().__init__(params, X, Z)


class DistributedGaussianInpaintingTvModel(
    BaseGaussianInpaintingTvModel,
    BaseDistributedModel,
):
    def __init__(
        self,
        params: GaussianInpaintingTvParameters,
        X: BaseTransitionKernel,
        Z: BaseTransitionKernel,
        comm: MPI.Comm,
        grid_size: np.ndarray,
        full_size: np.ndarray,
    ):
        self.comm = comm
        self.full_size = full_size

        self.gradient_operator = DistributedGradient2d(
            self.full_size, grid_size, self.comm
        )
        super().__init__(params, X, Z)

    def set_slices(self):
        """set_slices Describes which portion of the global buffer the current thread must handle.

        Returns
        -------
        dict
            Dictionary containing the slices of the global buffer that this thread will handle.
        """
        self.slices["X"] = (
            self.gradient_operator.direct_communicator.cartslicer.slice_global_buffer_to_tile
        )
        self.slices["Z"] = (
            np.s_[:],
            *self.gradient_operator.direct_communicator.cartslicer.slice_global_buffer_to_tile,
        )

    def set_global_sizes(self):
        """Describe the global sizes of several global buffers.

        Returns
        -------
        dict
            Global sizes of the variable of interest.
        """
        self.global_sizes["X"] = np.asarray(self.full_size, dtype=int)
        self.global_sizes["Z"] = np.asarray([2, *self.full_size], dtype=int)

    def set_local_sizes(self):
        self.local_sizes["X"] = self.X.current_state.shape
        self.local_sizes["Z"] = self.Z.current_state.shape
