r"""Test the extraction and insertion of the internal state of random number
generator in numpy.

NOTE
----
Tests may be broken with later numpy version, as :func:`mcmc.DataManager.warmstart_rng.extract_rng_state` and
:func:`mcmc.DataManager.warmstart_rng.restore_rng_state` rely on private methods
of :class:`numpy.random.Generator.bit_generator`.
"""

import numpy as np
from mcmc.DataManager.warmstart_rng import extract_rng_state, restore_rng_state

import pytest

pytestmark = pytest.mark.numpy


@pytest.fixture
def size():
    return 100000


def test_rng_state(size):
    r"""Test state extraction and reset with a ``numpy`` random number
    generator."""
    rng = np.random.default_rng(1234)
    saved_state, saved_inc = extract_rng_state(rng)
    a = rng.standard_normal(size)

    rng2 = np.random.default_rng(5678)
    restore_rng_state(rng2, saved_state, saved_inc)
    b = rng2.standard_normal(size)

    assert np.allclose(a, b)


if __name__ == "__main__":
    size = 100000
    test_rng_state(size)
