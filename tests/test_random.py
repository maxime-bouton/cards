import numpy as np
import pytest

from cards.random import create_rng, restore_rng, serialize_rng


@pytest.mark.cpu
def test_resume_cpu_rng(seed, ctx):
    rng = create_rng(seed, ctx)
    rng.random(1000)  # type: ignore

    schema = serialize_rng(rng)
    restored = restore_rng(schema)

    expected = rng.integers(0, 1_000_000, size=10)  # type: ignore
    actual = restored.integers(0, 1_000_000, size=10)  # type: ignore
    np.testing.assert_equal(expected, actual)


@pytest.mark.gpu
def test_resume_gpu_rng(seed, ctx):
    import torch

    rng = create_rng(seed, ctx)
    torch.rand(1000, generator=rng, device="cuda")  # type: ignore

    schema = serialize_rng(rng)
    restored = restore_rng(schema)

    assert rng.initial_seed() == restored.initial_seed()  # type: ignore
    assert rng.get_offset() == restored.get_offset()  # type: ignore

    expected = torch.rand(10, generator=rng, device="cuda")  # type: ignore
    actual = torch.rand(10, generator=restored, device="cuda")  # type: ignore
    assert torch.equal(expected, actual)


def test_restore_rejects_unrecognized_schema():
    with pytest.raises(ValueError):
        restore_rng({"any_key": np.zeros(1, dtype=np.uint8)})
