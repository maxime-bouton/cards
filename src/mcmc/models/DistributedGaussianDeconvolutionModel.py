r"""Implements a denoising model to solve a deconvolution problem under additive white Gaussian noise. Relies on ``numpy`` as a computing backend and MPI for the communications."""

import numpy as np

from mcmc.estimator.SerialMMSEBuilder import SerialMMSEBuilder
from mcmc.functionals.numpy.prox import l21_norm, prox_l21norm
from mcmc.models.BaseModel import BaseDistributedModel
from mcmc.distributed_operators.gradient import distributed_gradient2d
from mcmc.distributed_operators.sync_linear_convolution import SyncLinearConvolution
from mcmc.TransitionKernel.TransitionKernel import PSGLA

from mpi4py import MPI


def prox_nonegativity(x):
    return np.maximum(x, 0)


class DistributedGaussianDeconvolutionModel(BaseDistributedModel):
    def __init__(
        self,
        comm: MPI.Comm,
        full_size: np.ndarray,
        grid_size: np.ndarray,
        observations: np.ndarray,
        convolution_kernel: np.ndarray,
        X: PSGLA,
        Z: PSGLA,
        sigma2: float,
        reg_coeff: float,
        split_coeff: float,
    ) -> None:
        """
        Parameters
        ----------
        comm : MPI.Comm
            MPI communicator.
        full_size : np.ndarray
            Dimensions of the image.
        grid_size : np.ndarray
            Dimensions of the local subarrays.
        observations : np.ndarray
            Local tile of the deteriorated picture than can be observed.
        convolution_kernel : np.ndarray
            Convolution kernel associated to the convolution operator, expected to be a gaussian kernel.
        X : BaseSerialTransitionKernel
            Transition kernel for a subarray of the main variable.
        Z : BaseSerialTransitionKernel
            Transition kernel for a subarray of the splitting variable.
        sigma2 : float
            Standard deviation of the gausssian noise, expexted to be known.
        reg_coeff : float
            Regularisation coefficient.
        split_coeff : float
            Splitting coefficient.
        """
        self.comm = comm
        self.full_size = full_size
        self.observations = observations
        self.convolution_kernel = convolution_kernel
        self.X = X
        self.Z = Z
        self.reg_coeff = reg_coeff
        self.split_coeff = split_coeff
        self.sigma2 = sigma2

        self.estimator_builder = SerialMMSEBuilder(self.X.current_state.shape)

        self.gradient_handler = distributed_gradient2d(self.full_size, grid_size)
        self.adj_buffer = np.zeros(
            self.gradient_handler.cart_comm.cartslicer.tile_size
        )  #! buffer linked to the kernel used
        self.convolution_handler = SyncLinearConvolution(
            self.full_size, self.convolution_kernel, self.comm, grid_size
        )
        self.convolution_product = np.zeros(
            self.convolution_handler.adjoint_communicator.cartslicer.tile_size
        )

        self.rank = comm.Get_rank()

        M, N = self.gradient_handler.cart_comm.cartslicer.tile_size
        self.M = M
        self.N = N

        self.global_sizes = self.set_global_sizes()
        self.slices = self.set_slices()

        if type(X) is PSGLA:
            self.X.prox = prox_nonegativity
            self.X.grad = (
                lambda x: self.convolution_handler.adjoint(
                    self.convolution_product - self.observations
                )[: self.M, : self.N]  #! better way to crop?
                / self.sigma2
                + self.adj_buffer / self.split_coeff
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

        if type(Z) is PSGLA:
            self.Z.prox = lambda z: (
                prox_l21norm(z, lam=self.Z.step_size * self.reg_coeff)
            )
            self.Z.grad = lambda z: (z - self.gradX) / self.split_coeff
        else:
            raise ValueError("Kernel type not yet supported by this model.")

    def set_slices(self):
        slices = {}
        slices["X"] = (
            self.gradient_handler.cart_comm.cartslicer._get_slice_global_buffer_to_tile()
        )
        slices["MMSE"] = (
            self.gradient_handler.cart_comm.cartslicer._get_slice_global_buffer_to_tile()
        )
        slices["Z"] = (
            np.s_[:],
            *self.convolution_handler.direct_communicator.cartslicer._get_slice_global_buffer_to_tile(),
        )
        return slices

    def set_global_sizes(self) -> dict:
        sizes = {}
        sizes["X"] = np.asarray(self.full_size, dtype=int)
        sizes["Z"] = np.asarray(
            [
                2,
                *np.asarray(self.full_size),
            ],
            dtype=int,
        )
        sizes["MMSE"] = np.asarray(self.full_size, dtype=int)
        return sizes

    def get_states(self) -> dict:
        """get_states
        Extracts the current state of the transition kernel and other variables of interest and return it in a dictionnary.

        Returns
        -------
        dict
            Dictionnary containing the curent states of the variables.
        """
        states = {}
        states["X"] = self.X.current_state
        states["Z"] = self.Z.current_state
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
        self.X.current_state = states["X"].copy()
        self.Z.current_state = states["Z"].copy()

        self.gradX = self.gradient_handler.forward(self.X.current_state)
        self.convolution_product = self.convolution_handler.forward(
            self.X.current_state
        )

        self.gradient_handler.adjoint(
            self.adj_buffer,
            self.gradX[0] - self.Z.current_state[0],
            self.gradX[1] - self.Z.current_state[1],
        )

    def aggregate_states(self):
        self.estimator_builder.aggregate_states(self.X.current_state)

    def update(self, rng: np.random.Generator) -> None:
        """update Gobal update of the model. Updates every kernel used by the model and computes annex variables.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator, given by the sampler.
        """
        self.X.mc_step(rng)

        # update buffers related to X
        #! to do : factorize communications
        self.gradX = self.gradient_handler.forward(self.X.current_state)

        self.convolution_product = self.convolution_handler.forward(
            self.X.current_state
        )

        self.Z.mc_step(rng)

        #! kernel dependency
        self.gradient_handler.adjoint(
            self.adj_buffer,
            self.gradX[0] - self.Z.current_state[0],
            self.gradX[1] - self.Z.current_state[1],
        )

    def compute_potential(self) -> float:
        """compute_potential Compute the partial potential associated to the tile accessible by the worker process.

        Returns
        -------
        float
            Partial potential.
        """
        p = (0.5 / self.sigma2) * np.sum(
            (self.observations - self.convolution_product) ** 2
        )
        p += np.sum((self.gradX - self.Z.current_state) ** 2) * (0.5 / self.split_coeff)
        p += self.reg_coeff * l21_norm(self.Z.current_state)
        return p
