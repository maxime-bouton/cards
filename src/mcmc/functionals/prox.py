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


def KL(x: xp.ndarray, y: xp.ndarray) -> float:
    eps = xp.finfo(x.dtype).eps
    return xp.sum(x * xp.log(xp.maximum(x, eps) / xp.maximum(y, eps))) + xp.sum(y - x)


def prox_KL(x: xp.ndarray, y: xp.ndarray, lam: float = 1.0) -> xp.ndarray:
    x1 = x - lam
    return (x1 + xp.sqrt(x1**2 + 4 * lam * y)) / 2


def prox_l1norm(x, lam):
    if lam <= 0:
        raise ValueError("`lam` should be positive.")
    return xp.sign(x) * xp.maximum(xp.abs(x) - lam, 0)
