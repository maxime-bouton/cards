import cupy as cp
import numpy as np

from mcmc.estimator.GPUMMSEBuilder import GPUMMSEBuilder
from mcmc.models.BaseModel import BaseModel
from mcmc.operators.gpu.convolution import GpuConvolution
from mcmc.operators.gpu.gradient import GpuGradient2d
from mcmc.TransitionKernel.GpuTransitionKernel import BaseGpuTransitionKernel, GpuPSGLA
from mcmc.functionals.gpu.prox import prox_nonegativity, l21_norm, prox_l21norm


class GpuGaussianDeconvolutionModel(BaseModel):
    def __init__(
        self,
        observations: cp.ndarray,
        convolution_kernel: cp.ndarray,
        X: BaseGpuTransitionKernel,
        Z: BaseGpuTransitionKernel,
        sigma2: float,
        reg_coeff: float,
        split_coeff: float,
    ) -> None:
        """
        Parameters
        ----------
        observations : cp.ndarray
            Deteriorated picture than can be observed.
        convolution_kernel : cp.ndarray
            Convolution kernel associated to the convolution operator, expected to be a gaussian kernel.
        X : BaseGpuTransitionKernel
            Transition kernel for the main variable.
        Z : BaseGpuTransitionKernel
            Transition kernel for the splitting variable.
        sigma2 : float
            Standard deviation of the gausssian noise, expexted to be known.
        reg_coeff : float
            Regularisation coefficient.
        split_coeff : float
            Splitting coefficient.
        """
        self.observations = observations
        self.convolution_kernel = convolution_kernel
        self.X = X
        self.Z = Z
        self.reg_coeff = reg_coeff
        self.split_coeff = split_coeff
        self.sigma2 = sigma2

        self.estimator_builder = GPUMMSEBuilder(self.X.current_state.shape)
        self.gradient_operator = GpuGradient2d(
            cp.array([*self.X.current_state.shape], dtype=int)
        )
        # self.gradX = cp.zeros((2, *self.X.current_state.shape))
        self.gradX = self.gradient_operator.forward(self.X.current_state)

        M, N = self.X.current_state.shape  # picture dimensions
        m, n = self.convolution_kernel.shape  # convolution kernel dimensions
        self.M = M
        self.N = N
        self.convolution_handler = GpuConvolution(
            np.asarray([M, N]),
            self.convolution_kernel,
            (M + m - 1, N + n - 1),
        )
        # self.convolution_product = cp.zeros_like(observations)
        self.convolution_product = self.convolution_handler.forward(
            self.X.current_state
        )

        if type(X) is GpuPSGLA:
            self.X.prox = prox_nonegativity
            self.X.grad = (
                lambda x: self.convolution_handler.adjoint(
                    self.convolution_product - self.observations
                )[: self.M, : self.N]  #! better way to crop?
                / self.sigma2
                + self.gradient_operator.adjoint(self.gradX - self.Z.current_state)
                / self.split_coeff
            )
        else:
            raise ValueError("Kernel type not yet supported by this model.")

        if type(Z) is GpuPSGLA:
            self.Z.prox = lambda z: (prox_l21norm(z, self.Z.step_size * self.reg_coeff))
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
        states = {}
        states["X"] = cp.asnumpy(self.X.current_state)
        states["Z"] = cp.asnumpy(self.Z.current_state)
        states["MMSE"] = cp.asnumpy(self.estimator_builder.estimator)
        return states

    def set_states(self, states):
        """set_states
        Read the dictionnary given in entry and set the variables of the model to the values contained in it.
        The keys used by the dictionnary must be the same as in "get_states"

        Parameters
        ----------
        states : dict
            Dictionnary containing new values for the variables of the model.
        """

        self.X.current_state = cp.asarray(states["X"])
        self.Z.current_state = cp.asarray(states["Z"])

        self.gradX = self.gradient_operator.forward(self.X.current_state)
        self.convolution_product = self.convolution_handler.forward(
            self.X.current_state
        )
        return

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
        self.gradX = self.gradient_operator.forward(self.X.current_state)

        self.convolution_product = self.convolution_handler.forward(
            self.X.current_state
        )

        self.Z.mc_step(rng)

    def compute_potential(self) -> float:
        p = (0.5 / self.sigma2) * cp.sum(
            (self.observations - self.convolution_product) ** 2
        )
        p += cp.sum((self.gradX - self.Z.current_state) ** 2) * (0.5 / self.split_coeff)
        p += self.reg_coeff * l21_norm(self.Z.current_state)
        return p
