"""Implement a model used to build a solution to an inpainting problem under gaussian noise.
Can be executed on cpu or gpu depending on the settings of the backend.py file.
"""

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from mcmc.backend import gpu_context, xp
from mcmc.estimator.mmse_builder import mmse_builder, multi_gpu_mmse_builder
from mcmc.functionals.prox import l21_norm, prox_l21norm, prox_nonegativity
from mcmc.models.base_model import BaseModel
from mcmc.operators.gradient import Gradient2d
from mcmc.operators.linear_operator import LinearOperator
from mcmc.operators.mpi_gradient import MpiGradient2d
from mcmc.transition_kernel.base_transition_kernel import BaseTransitionKernel
from mcmc.transition_kernel.gpu_psgla import GpuPSGLA
from mcmc.transition_kernel.psgla import PSGLA


@dataclass
class InpaintingParameters:
    observations: xp.ndarray
    mask: xp.ndarray
    sigma2: float
    reg_coeff: float
    split_coeff: float


class BaseInpaintingModel(BaseModel):
    def __init__(
        self,
        params: InpaintingParameters,
        X: BaseTransitionKernel,
        Z: BaseTransitionKernel,
        gradient_op: LinearOperator,
        estimator_handler: mmse_builder,
        gpu_id: int = 0,
    ) -> None:
        self.X = X
        self.Z = Z
        self.reg_coeff = params.reg_coeff
        self.split_coeff = params.split_coeff
        self.sigma2 = params.sigma2
        self.gpu_id = gpu_id

        self.gradient_operator = gradient_op
        self.estimator_builder = estimator_handler

        with gpu_context(self.gpu_id):
            self.observations = params.observations
            self.mask = params.mask
            self.gradX = xp.zeros_like(self.X.current_state)

        self.set_conditionals()

    def set_conditionals(self) -> None:
        """Set the conditionals of the transition kernels including the coupling between those kernels."""
        with gpu_context(self.gpu_id):
            if (type(self.X) is PSGLA) or (type(self.X) is GpuPSGLA):
                self.X.prox = prox_nonegativity
                self.X.grad = (
                    lambda x: self.mask * (x - self.observations) / self.sigma2
                    + self.gradient_operator.adjoint(self.gradX - self.Z.current_state)
                    / self.split_coeff
                )
            else:
                raise ValueError("Kernel type not yet supported by this model.")

            if (type(self.Z) is PSGLA) or (type(self.Z) is GpuPSGLA):
                self.Z.prox = lambda z: (
                    prox_l21norm(z, lam=self.Z.step_size * self.reg_coeff, axis=0)
                )
                self.Z.grad = lambda z: (z - self.gradX) / self.split_coeff
            else:
                raise ValueError("Kernel type not yet supported by this model.")

    def get_states(self) -> dict:
        """get_states
        Extracts the current state of the transition kernel and other variables of interest and return the in a dictionnary.

        Returns
        -------
        dict
            Dictionnary containing the curent states of the variables.
        """
        # TODO to be abstracted
        states = {}
        with gpu_context(self.gpu_id):
            states["X"] = self.X.get_state()
            states["Z"] = self.Z.get_state()
        if type(self.X) is GpuPSGLA:
            with gpu_context(self.gpu_id):
                states["MMSE"] = self.estimator_builder.estimator.get()
        else:
            states["MMSE"] = self.estimator_builder.estimator
        return states

    def set_states(self, states: dict) -> None:
        """set_states
        Read the dictionnary given in entry and set the variables of the model to the values contained in it.
        The keys used by the dictionnary must be the same as in "get_states"

        Parameters
        ----------
        states : dict
            Dictionnary containing new values for the variables of the model.
        """
        with gpu_context(self.gpu_id):
            self.X.current_state = xp.asarray(states["X"])
            self.Z.current_state = xp.asarray(states["Z"])
            self.gradX = self.gradient_operator.forward(self.X.current_state)

    def update(self, rng: np.random.Generator) -> None:
        """update Gobal update of the model. Updates every kernel used by the model and computes annex variables.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator, given by the sampler.
        """
        self.X.mc_step(rng)
        self.gradX = self.gradient_operator.forward(self.X.current_state)
        self.Z.mc_step(rng)

    def aggregate_states(self):
        self.estimator_builder.aggregate_states(self.X.current_state)

    def compute_potential(self) -> float:
        """compute_potential Computes the potential.

        Returns
        -------
        float
            Potential of the targeted law.
        """
        with gpu_context(self.gpu_id):
            p = xp.sum((self.observations - self.mask * self.X.current_state) ** 2) / (
                2 * self.sigma2
            )
            p += xp.sum((self.gradX - self.Z.current_state) ** 2) / (
                2 * self.split_coeff
            )
            p += self.reg_coeff * l21_norm(self.Z.current_state)
            return p


class InpaintingModel(BaseInpaintingModel):
    def __init__(self, params: InpaintingParameters, X, Z, gpu_id=0):
        estimator_builder = mmse_builder(params.observations.shape)
        gradient_op = Gradient2d(np.array([*X.current_state.shape], dtype=int))

        super().__init__(params, X, Z, gradient_op, estimator_builder, gpu_id)
        return


class DistributedInpaintingModel(BaseInpaintingModel):
    def set_slices(self) -> dict:
        """set_slices Describes which portion of the global buffer the current thread must handle.

        Returns
        -------
        dict
            Dictionary containing the slices of the global buffer that this thread will handle.
        """
        slices = {}
        slices["X"] = (
            self.gradient_operator.cart_comm.cartslicer._get_slice_global_buffer_to_tile()
        )
        slices["Z"] = (
            np.s_[:],
            *self.gradient_operator.cart_comm.cartslicer._get_slice_global_buffer_to_tile(),
        )
        slices["MMSE"] = (
            self.gradient_operator.cart_comm.cartslicer._get_slice_global_buffer_to_tile()
        )
        self.slices = slices
        return

    def set_global_sizes(self) -> dict:
        """set_global_sizes Describe the gobla sizes of several global buffers.

        Returns
        -------
        dict
            Global sizes of the variable of interest.
        """
        sizes = {}
        sizes["X"] = np.asarray(self.full_size, dtype=int)
        sizes["Z"] = np.asarray([2, *self.full_size], dtype=int)
        sizes["MMSE"] = np.asarray(self.full_size, dtype=int)

        self.global_sizes = sizes
        return

    def __init__(
        self,
        comm: MPI.Comm,
        full_size: np.ndarray,
        grid_size: np.ndarray,
        params: InpaintingParameters,
        X,
        Z,
        gpu_id=0,
    ) -> None:
        self.comm = comm
        self.rank = comm.Get_rank()
        self.full_size = full_size
        self.gpu_id = gpu_id

        gradient_op = MpiGradient2d(
            np.asarray(self.full_size), grid_size, self.comm, gpu_id
        )
        estimator_builder = multi_gpu_mmse_builder(
            gradient_op.cart_comm.cartslicer.tile_size, gpu_id
        )

        super().__init__(params, X, Z, gradient_op, estimator_builder, self.gpu_id)

        self.slices = {}
        self.set_slices()
        self.global_sizes = {}
        self.set_global_sizes()
