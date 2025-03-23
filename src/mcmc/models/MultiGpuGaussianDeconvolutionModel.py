import numpy as np
import cupy as cp

from mcmc.distributed_operators.multi_gpu.gradient import distributed_gradient2d
from mcmc.distributed_operators.multi_gpu.dft_convolution import MultiGPU_DFTConvolution

from mcmc.estimator.GPUMMSEBuilder import MultiGpuMMSEBuilder
from mcmc.functionals.gpu.prox import l21_norm, prox_l21norm
from mcmc.models.BaseModel import BaseDistributedModel
from mcmc.TransitionKernel.GpuTransitionKernel import (
    MultiGpuPSGLA,
)

from mpi4py import MPI


def prox_nonegativity(x):
    return cp.maximum(x, 0)


class MultiGpuGaussianDeconvolutionModel(BaseDistributedModel):
    """
    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator.
    full_size : np.ndarray
        Dimensions of the image.
    grid_size : np.ndarray
        Dimensions of the local subarrays.
    observations : cp.ndarray
        Local tile of the deteriorated picture than can be observed.
    convolution_kernel : cp.ndarray
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

    def __init__(
        self,
        comm: MPI.Comm,
        full_size: np.ndarray,
        grid_size: np.ndarray,
        observations: cp.ndarray,
        convolution_kernel: cp.ndarray,
        X: MultiGpuPSGLA,
        Z: MultiGpuPSGLA,
        sigma2: float,
        reg_coeff: float,
        split_coeff: float,
    ) -> None:
        self.comm = comm
        self.rank = comm.Get_rank()
        self.full_size = full_size
        self.nb_gpu = cp.cuda.runtime.getDeviceCount()
        self.gpu_id = self.rank % self.nb_gpu

        self.X = X
        self.Z = Z
        self.reg_coeff = reg_coeff
        self.split_coeff = split_coeff
        self.sigma2 = sigma2

        with cp.cuda.Device(self.gpu_id):
            self.observations = cp.asarray(observations)
            self.convolution_kernel = cp.asarray(convolution_kernel)

        self.estimator_builder = MultiGpuMMSEBuilder(
            self.X.current_state.shape, self.gpu_id
        )

        self.gradient_handler = distributed_gradient2d(self.full_size, grid_size)
        self.convolution_handler = MultiGPU_DFTConvolution(
            self.full_size, self.convolution_kernel, self.comm, grid_size
        )

        with cp.cuda.Device(self.gpu_id):
            self.adj_buffer = cp.zeros(
                self.gradient_handler.cart_comm.cartslicer.tile_size
            )  #! buffer linked to the kernel used

            #self.convolution_product = cp.zeros(
            #    self.convolution_handler.adjoint_communicator.cartslicer.tile_size
            #)
            self.convolution_product = cp.zeros_like(self.observations)
            self.gradX = cp.zeros(
                (2, *self.gradient_handler.cart_comm.cartslicer.tile_size)
            )

        M, N = self.gradient_handler.cart_comm.cartslicer.tile_size
        self.M = M
        self.N = N

        assert(self.observations.shape == self.convolution_product.shape)

        self.global_sizes = self.set_global_sizes()
        self.slices = self.set_slices()

        if type(X) is MultiGpuPSGLA:
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

        if type(Z) is MultiGpuPSGLA:
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
        states["X"] = self.X.current_state.get()
        states["Z"] = self.Z.current_state.get()
        states["MMSE"] = self.estimator_builder.estimator.get()
        return states

    def aggregate_states(self):
        self.estimator_builder.aggregate_states(self.X.current_state)

    def set_states(self, states: dict) -> None:
        return NotImplemented

    def update(self, rng: np.random.Generator) -> None:
        """update Gobal update of the model. Updates every kernel used by the model and computes annex variables.

        Parameters
        ----------
        rng : torch.Generator
            Random number generator, given by the sampler.
        """
        # update buffers related to X
        #! to do : factorize communications
        with cp.cuda.Device(self.gpu_id):
            self.X.mc_step(rng)

            self.gradX = self.gradient_handler.forward(self.X.current_state)

            self.convolution_product = self.convolution_handler.forward(
                self.X.current_state
            )

            self.Z.mc_step(rng)

            #! kernel dependency
            self.adj_buffer = self.gradient_handler.adjoint(
                self.gradX[0] - self.Z.current_state[0],
                self.gradX[1] - self.Z.current_state[1],
            )  # can be rework to be called in the lambda that define the conditionnal
            

    def compute_potential(self) -> float:
        """compute_potential Compute the partial potential associated to the tile accessible by the worker process.

        Returns
        -------
        float
            Partial potential.
        """
        with cp.cuda.Device(self.gpu_id):
            p = (0.5 / self.sigma2) * cp.sum(
                (self.observations - self.convolution_product) ** 2
            )
            p += cp.sum((self.gradX - self.Z.current_state) ** 2) * (
                0.5 / self.split_coeff
            )
            p += self.reg_coeff * l21_norm(self.Z.current_state)
            return p
