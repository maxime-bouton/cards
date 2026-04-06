r"""Abstract CPU implementation for the Moreau-Yosida Unajusted Langevin Algorithm (MYULA) :cite:p:`Durmus2018`."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.
#
# adpated from: https://gitlab.cristal.univ-lille.fr/pthouven/dsgs

# TODO: documentation

from typing import Optional

from cards.backend import xp
from cards.transition_kernel.base_transition_kernel import BaseTransitionKernel


class MYULA(BaseTransitionKernel):
    r"""MYULA transition kernel.

    MYULA transition kernel :cite:p:`Durmus2018` associated with a target
    density of the form

    .. math::
        \pi(x) \propto \exp( -f(x) - g(x)),

    with :math:`f \in \Gamma_0 (\mathbb{R}^N)` an :math:`L_f`-smooth function
    and :math:`g \in \Gamma_0 (\mathbb{R}^N)`.

    Attributes
    ----------
    lipschitz_cst : float
        Lipschitz constant :math:`L_f` of the smooth function :math:`f`.
    step_size : float, optional
        Stepsize :math:`\gamma` for the kernel, statisfying
        :math:`0 < \gamma = s\lambda (L_f \lambda + 1)^{-1}`.
    reg_cst : float, optional
        Moreau-Yosida smoothing parameter :math:`\lambda = r L_f^{-1}`, by default 1.
    _cst1, _cst2 : float
        Precomputed auxiliary quantities involved in the MYULA update step.

    Note
    ----
    Starting form state :math:`x^{(k)}`, the MYULA transition is given by

    .. math ::
        x^{(k+1)} = \big( 1 - \frac{\gamma}{\lambda} \big) x^{(k)} - \gamma
        \nabla f (x^{(k)}) + \frac{\gamma}{\lambda} \text{prox}_{\gamma g}
        (x^{(k)}) + \sqrt{2 \gamma} \xi^{(k+1)},

    with :math:`x^{(k)}` the current state at iteration :math:`k`,
    :math:`\lambda > 0` the regularization parameter and
    :math:`\gamma \in \big(0, \lambda (L_f \lambda + 1)^{-1} \big]` the
    stepsize. Typical values for :math:`\lambda` and :math:`gamma` are :cite:p:`Durmus2018`

    .. math::
        \lambda = L_f^{-1}, or \lambda \in [5 L_f^{-1}, 10 L_f^{-1}],
        \quad
        \gamma \in \big[ \lambda \big( 5 (L_f\lambda + 1) \big)^{-1},
        \lambda (L_f\lambda + 1)^{-1} \big)
    """

    def __init__(
        self,
        state_shape,
        lipschitz_cst: float,
        regularization_factor: float = 1.0,
        stepsize_factor: float = 0.5,
        dtype: Optional[xp.dtype] = None,
        initial_value: xp.ndarray | None = None,
    ) -> None:
        r"""Constructor.

        Default recommendations from :cite:p:`Durmus2018` are used to set the
        stepsize and the regularization parameters involved in the MYULA
        transition kernel.

        Parameters
        ----------
        state_shape : tuple[int, ...]
            Shape of the parameter handled by the transition kernel.
        lipschitz_cst : float
            Lipschitz constant :math:`L_f` of :math:`\nabla f`.
        regularization_factor : float, optional
            Multiplicative factor :math:`r \in (0, 1]` to adjust the
            Moreau-Yosida smoothing parameter :math:`\lambda = r L_f^{-1}` of
            :math:`g`, by default 1.0
        stepsize_factor : float, optional
            Stepsize factor :math:`s` reducing the default stepsize adopted for
            MYULA, by default 0.5
        dtype : xp.dtype | None, optional
            Parameter type, by default None.
        initial_value: xp.ndarray | None, optional
            Initial parameter value, by default None.

        Raises
        ------
        ValueError
            The parameters ``stepsize_factor`` and ``regularization_factor``
            both need to be <= 1.
        """
        super(MYULA, self).__init__(
            state_shape, dtype=dtype, initial_value=initial_value
        )

        if xp.minimum(stepsize_factor, regularization_factor) > 1:
            raise ValueError(
                "`stepsize_factor` and `regularization_factor` both need to be <= 1."
            )

        self.lipschitz_cst = lipschitz_cst

        self.reg_cst = regularization_factor / lipschitz_cst
        self.step_size = stepsize_factor / (lipschitz_cst + 1 / self.reg_cst)

        self._cst1 = (2 * self.step_size) ** 0.5
        self._cst2 = self.step_size / self.reg_cst

    def set_params(
        self, new_regularization_factor: float = 1, new_stepsize_factor: float = 0.5
    ) -> None:
        r"""Set value of the regularization and stepsize factors.

        Parameters
        ----------
        new_regularization_factor : float
            New regularization parameter, by default 1.0
        new_stepsize_factor : float
            New stepsize parameter, by default 0.5
        """
        assert new_regularization_factor <= 1
        assert new_stepsize_factor <= 1

        self.reg_cst = new_regularization_factor / self.lipschitz_cst
        self.step_size = new_stepsize_factor / (self.lipschitz_cst + 1 / self.reg_cst)

        self._cst1 = (2 * self.step_size) ** 0.5
        self._cst2 = self.step_size / self.reg_cst

    # FIXME: keep regularization_factor and stepsize factor, to avoid unexpected reset to default stepsize and regularization values
    def set_lipschitz(
        self,
        new_lipschitz_cst: float,
        regularization_factor: float = 1,
        stepsize_factor: float = 0.5,
    ) -> None:
        r"""Update the value of the Lipschitz constant underlying the MYULA
        transition.

        Parameters
        ----------
        new_lipschitz_cst : float
            New value for the Lipschitz constant.
        """
        self.lipschitz_cst = new_lipschitz_cst
        self.set_params(regularization_factor, stepsize_factor)

    def prox(self, state: xp.ndarray) -> xp.ndarray:
        """Compute the proximity operator of the non-smooth term in the
        negative log-posterior function. Needs to be defined by the user."""
        raise ValueError("Warning : proximal operator has not be defined !")

    def grad(self, state: xp.ndarray) -> xp.ndarray:
        """Compute the gradient of the differentiable term in the negative
        log-posterior function. Needs to be defined by the user."""
        raise ValueError("Warning : gradient function has not be defined !")

    def mc_step(self, rng) -> None:
        self.current_state = (
            (1 - self._cst2) * self.current_state
            - self.step_size * self.grad(self.current_state)
            + self._cst2 * self.prox(self.current_state)
            + self._cst1
            * rng.standard_normal(self.current_state.shape, dtype=self.dtype)
        )
