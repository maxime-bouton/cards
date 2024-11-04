r"""Test the extraction and insertion of the internal state of random number
generator in numpy.

NOTE
----
May be unstable due to the use a private method of
``numpy.random.Generator.bit_generator``.
"""

import numpy as np
import sys

import pytest


@pytest.fixture
def size():
    return 10000000


def test_rng_state(size):
    N = size

    rng = np.random.default_rng(1234)
    new_state_state = rng.bit_generator.__getstate__()[0]["state"]["state"]
    new_state_inc = rng.bit_generator.__getstate__()[0]["state"]["inc"]

    a = rng.standard_normal(N)

    loaded_state_state = np.array(
        bytearray(new_state_state.to_bytes(32, sys.byteorder))
    )
    loaded_state_inc = np.array(bytearray(new_state_inc.to_bytes(32, sys.byteorder)))

    rng2 = np.random.default_rng(5678)
    new_state = rng2.bit_generator.__getstate__()
    new_state[0]["state"]["state"] = int.from_bytes(loaded_state_state, sys.byteorder)
    new_state[0]["state"]["inc"] = int.from_bytes(loaded_state_inc, sys.byteorder)

    rng2.bit_generator.__setstate__(new_state)

    b = rng2.standard_normal(N)

    assert np.allclose(a, b)


if __name__ == "__main__":
    size = 10000000
    test_rng_state(size)
