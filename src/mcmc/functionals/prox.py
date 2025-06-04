"""Implementation of useful mathematical functions such as simple proximal operators._"""

from mcmc.backend import xp


def prox_nonegativity(x):
    return xp.maximum(x, 0)


def l21_norm(x, axis=0):
    return xp.sum(xp.sqrt(xp.sum(x**2, axis=axis)))


def prox_l21norm(x, lam=1.0, axis=0):
    if lam <= 0:
        raise ValueError("`lam` should be positive.")
    return x * (1 - 1 / xp.maximum(xp.sqrt(xp.sum(x**2, axis=axis)) / lam, 1.0))
