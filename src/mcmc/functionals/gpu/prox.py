import cupy as cp


def prox_nonegativity(x):
    return cp.maximum(x, 0)


def l21_norm(x, axis=0):
    return cp.sum(cp.sqrt(cp.sum(x**2, axis=axis)))


def prox_l21norm(x, lam=1.0, axis=0):
    if lam <= 0:
        raise ValueError("`lam` should be positive.")
    return x * (1 - 1 / cp.maximum(cp.sqrt(cp.sum(x**2, axis=axis)) / lam, 1.0))
