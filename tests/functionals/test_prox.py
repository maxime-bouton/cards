r"""Basic tests for the few mathematical functions and associated proximal operators implemented."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

import pytest

import cards.backend as xp
from cards.functionals import prox


def test_prox_nonegativity(seed, input_shape):
    rng = xp.random.default_rng(seed)
    x = rng.random(input_shape)
    z = prox.prox_nonegativity(x)
    xp.testing.assert_allclose(z[z > 0], x[x > 0])


def test_KL(input_shape, seed):
    rng = xp.random.default_rng(seed)
    x = rng.random(input_shape)
    z = prox.KL(x, xp.zeros_like(x))
    eps = xp.finfo(x.dtype).eps
    xp.testing.assert_allclose(
        z, xp.sum(x * xp.log(xp.maximum(x, eps) / eps)) - xp.sum(x)
    )


def test_prox_KL(input_shape):
    z = prox.prox_KL(xp.zeros(input_shape), xp.zeros(input_shape), lam=1)
    xp.testing.assert_allclose(z, 0)


def test_prox_KL_negative_regularization_parameter(input_shape):
    with pytest.raises(ValueError) as excinfo:
        prox.prox_KL(xp.zeros(input_shape), xp.zeros(input_shape), lam=-1)
    assert "`lam` should be positive." in str(excinfo.value)


def test_l21_norm(input_shape, seed):
    rng = xp.random.default_rng(seed)
    x = rng.random(input_shape)
    z = prox.l21_norm(x, axis=0)
    xp.testing.assert_allclose(z, xp.sum(xp.sqrt(xp.sum(x**2, axis=0))))


def test_prox_l21norm(input_shape):
    z = prox.prox_l21norm(xp.zeros(input_shape), lam=1)
    xp.testing.assert_allclose(z, 0)


def test_prox_l21norm_negative_regularization_parameter(input_shape):
    with pytest.raises(ValueError) as excinfo:
        prox.prox_l21norm(xp.zeros(input_shape), lam=-1)
    assert "`lam` should be positive." in str(excinfo.value)
