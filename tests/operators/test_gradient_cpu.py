import numpy as np
import pytest

from mcmc.operators.gradient import Gradient2d  # backend ste to numpy by default


@pytest.fixture
def seed():
    return 1234


@pytest.fixture
def dims():
    return np.array([100, 75], "i")


def test_basic_check(dims):
    X = np.ones(dims)
    gradient_operator = Gradient2d(dims)

    grad = gradient_operator.forward(X)

    assert np.amax(np.abs(grad)) == 0


def test_adjacency_cpu(seed, dims):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(dims)
    Y = rng.standard_normal(np.asarray((2, *dims)))

    gradient_operator = Gradient2d(dims)

    Hx = gradient_operator.forward(X)
    Hy = gradient_operator.adjoint(Y)

    xHy = np.sum(X * Hy)
    Hxy = np.sum(Hx * Y)

    assert isinstance(Hx, np.ndarray)
    assert isinstance(Hy, np.ndarray)
    assert np.isclose(Hxy, xHy, atol=1e-15)


if __name__ == "__main__":
    default_seed = 1234
    default_dims = np.asarray([100, 75])

    test_basic_check(default_dims)
    test_adjacency_cpu(default_seed, default_dims)
