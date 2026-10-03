"""Implementation of the SK-ROCK transition kernel :cite:p:`Pereyra2020`."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

# TODO: to be adapted to current interface

# from typing import Callable

# import numpy as np
# from dsgs.experimental.transitions.base_transition_kernel import BaseTransitionKernel

# # TODO: see how to plug the gradient of a function without using lambdas (bad
# # TODO: practice)

# # TODO: set default value for smoothing parameter lambd

# # see how to use the Chebyshev class (not 100% obvious -> see in C++ as well in
# # the longer term)
# # https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.Chebyshev.html#numpy.polynomial.chebyshev.Chebyshev

# # ? - see how to define a staticmethod after class instantiation (avoid defining element as abstract)


# def set_skrock_parameters(
#     lipschitz_constant: float, lambd: float, degree: int, eta: float = 0.05
# ):
#     r"""Set the number of steps :math:`n` and maximum stepsize
#     :math:`\delta_{\max}` involved in the SK-ROCK transition kernel
#     :cite:p:`Pereyra2020`.

#     Parameters
#     ----------
#     lipschitz_constant : float
#         Lipschitz constant :math:`L_f` of the smooth function :math:`f`
#         involved in the target density :math:`\pi`.
#     lambd : float
#         Smoothing parameter :math:`\lambda` controlling the quality of the
#         smooth approximation :math:`\pi_{\lambda}` to the target density
#         :math:`\pi`.
#     degree : int, optional
#         Order of the Chebyshev polynomial of the 1st kind used in SK-ROCK.
#     eta : float, optional
#         Auxiliary parameter :math:`\eta` involved in SK-ROCK. By default 0.05.

#     Returns
#     -------
#     ls : float
#         Auxiliary step-size quantity involved in SK-ROCK :cite:p:`Pereyra2020`.
#     max_stepsize : float
#         Maximum stepsize for which convergence of the kernel can be ensured.
#     omega : numpy.ndarray
#         Vector containing the value for the parameters :math:`\omega_0` and
#         :math:`\omega_1`.
#     mu1 : float
#         Auxiliary parameter.
#     nu1 : float
#         Auxiliary parameter.
#     k1 : float
#         Auxiliary parameter.
#     """
#     ls = int(((degree - 0.5) ** 2) * (2 - 4 * eta / 3) - 1.5)
#     max_stepsize = ls / (lipschitz_constant + 1 / lambd)

#     cheb = np.polynomial.chebyshev.Chebyshev.basis(degree)
#     diff_cheb = cheb.deriv(1)
#     omega = np.empty((2,), dtype="d")
#     omega[0] = 1 + (eta / degree**2)
#     omega[1] = cheb(omega[0]) / diff_cheb(omega[0])

#     mu1 = omega[1] / omega[0]
#     nu1 = degree * omega[1] / 2
#     k1 = degree * mu1

#     return ls, max_stepsize, omega, mu1, nu1, k1


# class SKROCK(BaseTransitionKernel):
#     r"""Generic implementation of the SK-ROCK transition kernel
#     :cite:p:`Pereyra2020` to sample approximately sample from a target density
#     of the form

#     .. math::
#         \pi(x) \propto \exp( -f(x) - g(x)),

#     with :math:`f \in \Gamma_0 (\mathbb{R}^N)` an :math:`L_f`-smooth function
#     and :math:`g \in \Gamma_0 (\mathbb{R}^N)`.

#     Attributes
#     ----------
#     _eta : float
#         Auxiliary parameter invovled in SK-ROCK (fixed to 0.05).
#     lipschitz_constant : float
#         Lipschitz constant :math:`L_f` of the smooth function :math:`f`
#         involved in the target density :math:`\pi`.
#     _lambd : float
#         Smoothing parameter :math:`\lambda` controlling the quality of the
#         smooth approximation :math:`\pi_\lambda` to the target density
#         :math:`\pi`.
#     _stepsize_factor : float, optional
#         Multiplicative factor :math:`s \in (0, 1]` to adjust the stepsize
#         :math:`\gamma`, by default 0.99. The resulting stepsize statisfies
#         :math:`0 < \delta \leq n (L_f + \lambda^{-1} )^{-1}`, with
#         :math:`n` the number of steps considered in the kernel.
#     _degree : int, optional
#         Order of the Chebyshev polynomial of the 1st kind used in SK-ROCK
#         :cite:p:`Pereyra2020`, by default 10.
#     _eta : float, optional
#         Auxiliary parameter :math:`\eta` involved in SK-ROCK. By default 0.05.
#     ls : float
#         Auxiliary step-size quantity involved in SK-ROCK :cite:p:`Pereyra2020`.
#     max_stepsize : float
#         Maximum stepsize for which convergence of the kernel can be ensured.
#     omega : numpy.ndarray
#         Vector containing the value for the parameters :math:`\omega_0` and
#         :math:`\omega_1`.
#     mu1 : float
#         Auxiliary parameter.
#     nu1 : float
#         Auxiliary parameter.
#     k1 : float
#         Auxiliary parameter.
#     _stepsize : float
#         Step-size value adopted in the kernel.
#     _prox_scale : float, optional
#         Parameter of the proximal operator involved in the update steps, by
#         default 1.0.
#     _gradient_function : numpy.ndarray
#         Gradient of the smooth part of the potential.
#     _proximal_operator : function
#         In-place function implementing the proximal operator of the
#         non-smooth part of the potential (must admit a keyword argument
#         ``lam`` for a multiplicative constant).
#     _buffer_state : numpy.ndarray
#         Temporary buffer storing previous states, required in the integrator
#         used in SKROCK.
#     """

#     def __init__(
#         self,
#         state,
#         variable_name: str,
#         lipschitz_constant: float,
#         lambd: float,
#         gradient_function: Callable[[np.ndarray], np.ndarray],
#         proximal_operator: Callable[[np.ndarray, float], np.ndarray],
#         degree: int = 10,
#         eta: float = 0.05,
#         stepsize_factor: float = 1.0,
#         prox_scale: float = 1.0,
#     ):
#         r"""Constructor of the SKROCK kernel.

#         Parameters
#         ----------
#         state : numpy.ndarray
#             Initial state of the transition kernel.
#         variable_name : str
#             Name of the variable handled by the kernel.
#         lipschitz_constant : float
#             Lipschitz constant :math:`L_f` of the smooth function :math:`f`
#             involved in the target density :math:`\pi`.
#         lambd : float
#             Smoothing parameter :math:`\lambda` controlling the quality of the
#             smooth approximation :math:`\pi_\lambda` to the target density
#             :math:`\pi`. Following :cite:p:`Pereyra2022`, should be selected in
#             :math:`[L_f^{-1}, 10 L_f^{-1}]`.
#         gradient_function: (np.ndarray) -> np.ndarray
#             Function to evaluate :math:`\nabla f`, the gradient of the smooth
#             part of the potential.
#         proximal_operator: (np.ndarray, float) -> np.ndarray
#             Function to evaluate :math:`\text{prox}_{\beta g}`, the proximity
#             operator of the non-smooth part of the potential.
#         degree : int, optional
#             Order of the Chebyshev polynomial of the 1st kind used in SK-ROCK
#             :cite:p:`Pereyra2020`, by default 10.
#         eta : float, optional
#             Auxiliary parameter :math:`\eta` involved in SK-ROCK. By default 0.05.
#         stepsize_factor : float, optional
#             Multiplicative factor :math:`s \in (0, 1]` to adjust the stepsize
#             :math:`\gamma`, by default 1. The resulting stepsize statisfies
#             :math:`0 < \delta \leq n (L_f + \lambda^{-1} )^{-1}`, with
#             :math:`n` the number of steps considered in the kernel.
#         prox_scale : float, optional
#             Parameter of the proximal operator involved in the update steps. By
#             default 1.0

#         Raises
#         ------
#         ValueError
#             The Lipschitz constant needs to be positive.
#         ValueError
#             The smoothing parameter needs to be positive.
#         ValueError
#             `stepsize_factor` needs to be in (0, 1].
#         ValueError
#             ``prox_scale`` needs to be positive.
#         """
#         if lipschitz_constant <= 0:
#             raise ValueError(r"The Lipschitz constant needs to be positive.")
#         if lambd <= 0:
#             raise ValueError(r"The smoothing parameter needs to be positive.")
#         if stepsize_factor > 1 or stepsize_factor <= 0.0:
#             raise ValueError(r"`stepsize_factor` needs to be in (0, 1].")
#         if prox_scale <= 0:
#             raise ValueError(r"`prox_scale` needs to be positive.")

#         super(SKROCK, self).__init__(state, variable_name)
#         self._lipschitz_constant = lipschitz_constant
#         self._lambd = lambd
#         self._stepsize_factor = stepsize_factor
#         self._prox_scale = prox_scale
#         self._eta = eta
#         self._degree = degree

#         (
#             self.ls,
#             self.max_stepsize,
#             self.omega,
#             self.mu1,
#             self.nu1,
#             self.k1,
#         ) = set_skrock_parameters(
#             self._lipschitz_constant,
#             self._lambd,
#             degree=self._degree,
#             eta=self._eta,
#         )
#         self._stepsize = self._stepsize_factor * self.max_stepsize

#         self._gradient_function = gradient_function
#         self._proximal_operator = proximal_operator

#         # buffer to store intermediate states in the transition kernel
#         self._buffer_state = np.zeros(
#             (3, *self._current_state.shape), dtype=self._current_state.dtype
#         )

#     def reset_parameters(
#         self,
#         lipschitz_constant: float,
#         lambd: float,
#         degree: int = 10,
#         stepsize_factor: float = 1.0,
#         prox_scale: float = 1.0,
#     ):
#         r"""Reset base parameters of the kernel to any specified value. To be
#         used with care.

#         Parameters
#         ----------
#         lipschitz_constant : float
#             Lipschitz constant :math:`L_f` of the smooth function :math:`f`
#             involved in the target density :math:`\pi`.
#         lambd : float
#             Smoothing parameter :math:`\lambda` controlling the quality of the
#             smooth approximation :math:`\pi_\lambda` to the target density
#             :math:`\pi`.
#         degree : int, optional
#             Order of the Chebyshev polynomial of the 1st kind used in SK-ROCK
#             :cite:p:`Pereyra2020`, by default 10.
#         stepsize_factor : float, optional
#             Multiplicative factor :math:`s \in (0, 1]` to adjust the stepsize
#             :math:`\gamma`, by default 1. The resulting stepsize statisfies
#             :math:`0 < \delta \leq n (L_f + \lambda^{-1} )^{-1}`, with
#             :math:`n` the number of steps considered in the kernel.
#         prox_scale : float, optional
#             Parameter of the proximal operator involved in the update steps. by
#             default 1.0

#         Raises
#         ------
#         ValueError
#             The Lipschitz constant needs to be positive.
#         ValueError
#             The smoothing parameter needs to be positive.
#         ValueError
#             `stepsize_factor` needs to be in (0, 1].
#         ValueError
#             ``prox_scale`` needs to be positive.
#         """
#         if lipschitz_constant <= 0:
#             raise ValueError(r"The Lipschitz constant needs to be positive.")
#         if lambd <= 0:
#             raise ValueError(r"The smoothing parameter needs to be positive.")
#         if stepsize_factor > 1 or stepsize_factor <= 0.0:
#             raise ValueError(r"`stepsize_factor` needs to be in (0, 1].")
#         if prox_scale <= 0:
#             raise ValueError(r"`prox_scale` needs to be positive.")

#         self._lipschitz_constant = lipschitz_constant
#         self._lambd = lambd
#         self._degree = degree
#         self._stepsize_factor = stepsize_factor
#         self._prox_scale = prox_scale

#         (
#             self.ls,
#             self.max_stepsize,
#             self.omega,
#             self.nu1,
#             self.mu1,
#             self.k1,
#         ) = set_skrock_parameters(
#             lipschitz_constant,
#             lambd,
#             degree=self._degree,
#             eta=self._eta,
#         )
#         self._stepsize = self._stepsize_factor * self.max_stepsize

#     # ! need def + test for the value of prox_scale (> 0)
#     # ! ideally, keep this as a parameter of the prox object considered, to be
#     # plugged when instantiating the kernel considered
#     def _gradient_step(self, current_state):
#         r"""Evaluate the gradient of the smoothed potential involved in the
#         approximate density :math:`\pi_\lambda`.

#         For a target density :math:`\pi \propto \exp (-f - g)`, with
#         :math:`f \in \Gamma_0(\mathbb{R}^N)` an :math:`L_f`-smooth function and
#         :math:`g \in \Gamma_0(\mathbb{R}^N)`, one has

#             .. math::
#                 \nabla \log \pi_{\lambda} = - \nabla f(x) - \frac{1}{\lambda}
#                 \big( x - \text{prox}_{\lambda g} (x) \big).

#         Parameters
#         ----------
#         current_state : numpy.ndarray
#             Value at which the gradient needs to be evaluated

#         Returns
#         -------
#         numpy.ndarray
#             Gradient of the smoothed log-potential
#             :math:`\nabla \log \pi_{\lambda}` evaluated in ``current_state``.
#         """
#         gradient = (
#             -self._gradient_function(current_state)
#             - (
#                 current_state
#                 - self._proximal_operator(
#                     current_state, lam=self._lambd * self._prox_scale
#                 )
#             )
#             / self._lambd
#         )

#         return gradient

#     def _transition(self, rng):
#         """Implementation of the SK-ROCK probability transition kernel
#         described in :cite:p:`Pereyra2020`, Algorithm 1.

#         Parameters
#         ----------
#         rng : numpy.random.Generator
#             Random number generator.
#         """
#         # stochastic perturbation involved in the kernel
#         z = np.sqrt(2 * self._stepsize) * rng.standard_normal(
#             size=self._current_state.shape
#         )

#         # circular buffer to save the intermediate states
#         self._buffer_state[0] = self._current_state
#         self._buffer_state[1] = (
#             self._current_state
#             + self.mu1
#             * self._stepsize
#             * self._gradient_step(self._current_state + self.nu1 * z)
#             + self.k1 * z
#         )

#         # SK-ROCK implementation
#         for step in range(2, self._degree + 1):
#             ratio = np.polynomial.chebyshev.Chebyshev.basis(step - 1)(
#                 self.omega[0]
#             ) / np.polynomial.chebyshev.Chebyshev.basis(step)(self.omega[0])

#             mu = 2 * self.omega[1] * ratio
#             nu = 2 * self.omega[0] * ratio
#             k = 1 - nu
#             self._buffer_state[step % 3] = (
#                 mu
#                 * self._stepsize
#                 * self._gradient_step(self._buffer_state[(step - 1) % 3])
#                 + nu * self._buffer_state[(step - 1) % 3]
#                 + k * self._buffer_state[(step - 2) % 3]
#             )

#         # ! need to make sure there is a copy here?
#         self._current_state = self._buffer_state[self._degree % 3]
