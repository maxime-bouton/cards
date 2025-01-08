r"""Implements a denoising model to solve a deconvolution problem under additive white Gaussian noise. Relies on ``numpy`` as a computing backend."""

import numpy as np

from mcmc.estimator.SerialMMSEBuilder import SerialMMSEBuilder
from mcmc.functionals.numpy.prox import l21_norm, prox_l21norm
from mcmc.models.BaseModel import BaseModel
from mcmc.operators.gradient import Gradient2d
from mcmc.operators.serial_convolution import SerialConvolution
from mcmc.TransitionKernel.TransitionKernel import PSGLA


# FIXME: auxiliary functions must be defined elsewhere
def prox_nonegativity(x):
    return np.maximum(x, 0)


class GaussianDeconvolutionModel(BaseModel):
    def __init__(
        self,
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
        observations : np.ndarray
            Deteriorated picture than can be observed.
        convolution_kernel : np.ndarray
            Convolution kernel associated to the convolution operator, expected to be a gaussian kernel.
        X : BaseSerialTransitionKernel
            Transition kernel for the main variable.
        Z : BaseSerialTransitionKernel
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

        self.estimator_builder = SerialMMSEBuilder(self.X.current_state.shape)

        self.gradient_operator = Gradient2d(
            np.array([*self.X.current_state.shape], dtype=int)
        )
        # self.gradX = np.zeros((2, *self.X.current_state.shape))
        self.gradX = self.gradient_operator.forward(self.X.current_state)

        M, N = self.X.current_state.shape  # image dimensions
        m, n = self.convolution_kernel.shape  # convolution kernel dimensions
        self.M = M
        self.N = N
        self.convolution_handler = SerialConvolution(
            np.asarray([M, N]),
            self.convolution_kernel,
            np.asarray([M + m - 1, N + n - 1]),
        )
        # self.convolution_product = np.zeros_like(observations)
        self.convolution_product = self.convolution_handler.forward(
            self.X.current_state
        )

        if type(X) is PSGLA:
            self.X.prox = prox_nonegativity
            self.X.grad = (
                lambda x: self.convolution_handler.adjoint(
                    self.convolution_product - self.observations
                )[: self.M, : self.N]  #! better way to crop?
                / self.sigma2
                + self.gradient_operator.adjoint(self.gradX - self.Z.current_state)
                / self.split_coeff
                #! what is captured? self?
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

    def get_states(self) -> dict:
        """get_states
        Extracts the current state of the transition kernel and other variables of interest and return the in a dictionnary.

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

    def set_states(self, states):
        """set_states
        Read the dictionnary given in entry and set the variables of the model to the values contained in it.
        The keys used by the dictionnary must be the same as in "get_states"

        Parameters
        ----------
        states : dict
            Dictionnary containing new values for the variables of the model.
        """

        self.X.current_state = states["X"]
        self.Z.current_state = states["Z"]

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

        # update cached buffer related to X
        self.gradX = self.gradient_operator.forward(self.X.current_state)

        self.convolution_product = self.convolution_handler.forward(
            self.X.current_state
        )

        self.Z.mc_step(rng)

    def compute_potential(self) -> float:
        p = (0.5 / self.sigma2) * np.sum(
            (self.observations - self.convolution_product) ** 2
        )
        p += np.sum((self.gradX - self.Z.current_state) ** 2) * (0.5 / self.split_coeff)
        p += self.reg_coeff * l21_norm(self.Z.current_state)
        return p
