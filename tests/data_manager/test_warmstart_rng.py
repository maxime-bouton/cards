r"""Testing numpy and torch warmstart options for random number generators. States are saved to and loaded from a ``.h5`` file."""

from os.path import join

import h5py
import numpy as np
import pytest
import torch

import cards.backend as xp
from cards.io.warmstart_rng import (
    extract_rng_state,
    load_rng_np,
    load_rng_offset_torch,
    load_rng_torch,
    restore_rng_state,
    save_rng_np,
    save_rng_offset_torch,
    save_rng_torch,
)


@pytest.fixture
def sample_shape():
    return (1000,)


@pytest.fixture
def n_trials():
    return 3


@pytest.mark.serial
@pytest.mark.cpu
def test_warmstart_rng_np(tmp_path, seed, seed2, sample_shape):
    r"""Test warmstart of a numpy random number generator by restoring its
    state."""
    if tmp_path is not None:
        tmp_path_str = tmp_path.as_posix()
    else:
        tmp_path_str = ""
    filename = join(tmp_path_str, "warmstart_numpy_rng.h5")
    rng = np.random.default_rng(seed)

    x = rng.standard_normal(sample_shape)
    assert np.linalg.norm(x) > 0

    with h5py.File(filename, "w") as f:
        save_rng_np(rng, f)

    y = rng.standard_normal(sample_shape)

    new_rng = np.random.default_rng(seed2)
    with h5py.File(filename, "r") as f:
        load_rng_np(new_rng, f)

    z = new_rng.standard_normal(sample_shape)

    assert np.allclose(y, z)


@pytest.mark.serial
@pytest.mark.gpu
def test_warmstart_rng_offset_torch(tmp_path, seed, seed2, sample_shape, torch_device):
    r"""Test warmstart of a torch random number generator using the offset from
    an initial seed.
    """
    if tmp_path is not None:
        tmp_path_str = tmp_path.as_posix()
    else:
        tmp_path_str = ""
    filename = join(tmp_path_str, "warmstart_torch_rng.h5")

    rng = torch.Generator(device=torch_device).manual_seed(seed)

    x = torch.randn(sample_shape, generator=rng, device=torch_device)
    assert torch.linalg.vector_norm(x) > 0

    with h5py.File(filename, "w") as f:
        save_rng_offset_torch(rng, seed, f)

    y = torch.randn(sample_shape, generator=rng, device=torch_device)

    new_rng = torch.Generator(device=torch_device).manual_seed(seed2)
    with h5py.File(filename, "r") as f:
        load_rng_offset_torch(new_rng, f)

    z = torch.randn(sample_shape, generator=new_rng, device=torch_device)
    assert torch.allclose(y, z)


@pytest.mark.serial
@pytest.mark.gpu
def test_warmstart_rng_torch(tmp_path, seed, seed2, sample_shape, torch_device):
    r"""Test warmstart of a torch random number generator by restoring its
    state."""
    if tmp_path is not None:
        tmp_path_str = tmp_path.as_posix()
    else:
        tmp_path_str = ""
    filename = join(tmp_path_str, "warmstart_torch_rng.h5")

    rng = torch.Generator(device=torch_device).manual_seed(seed)

    x = torch.randn(sample_shape, generator=rng, device=torch_device)
    assert torch.linalg.vector_norm(x) > 0

    with h5py.File(filename, "w") as f:
        save_rng_torch(rng, f)

    y = torch.randn(sample_shape, generator=rng, device=torch_device)

    new_rng = torch.Generator(device=torch_device).manual_seed(seed2)
    with h5py.File(filename, "r") as f:
        load_rng_torch(new_rng, f)

    z = torch.randn(sample_shape, generator=new_rng, device=torch_device)
    assert torch.allclose(y, z)


@pytest.mark.serial
@pytest.mark.cpu
def test_rng_state(sample_shape: tuple[int, ...]):
    """Test state extraction and reset with a ``numpy`` random number
    generator."""
    rng = xp.random.default_rng(1234)
    _ = rng.standard_normal(sample_shape)
    saved_state, saved_inc = extract_rng_state(rng)
    a = rng.standard_normal(sample_shape)

    rng2 = xp.random.default_rng(5678)
    restore_rng_state(rng2, saved_state, saved_inc)
    b = rng2.standard_normal(sample_shape)

    xp.testing.assert_allclose(a, b)


# NOTE: this is about torch API testing, not about the library testing
@pytest.mark.serial
@pytest.mark.gpu
def test_restart_gpu_rng(sample_shape, seed, n_trials, torch_device):
    """Test the restart of a torch random number generator on GPU by saving
    and restoring its state.
    """
    rng = torch.Generator(device=torch_device).manual_seed(seed)

    for _ in range(n_trials):
        torch.normal(
            mean=0,
            std=1,
            size=sample_shape,
            generator=rng,
            device=torch_device,
        )

    rng2 = torch.Generator(device=torch_device).manual_seed(seed)
    # TODO: new interface since torch=2.9: .set_state()/.get_state() on CPU, .graphsafe_get_state()/graphsafe_set_state() on GPU
    # see: https://docs.pytorch.org/docs/2.9/generated/torch.Generator.html#torch.Generator
    # if device.type == "cpu":
    #     state = rng.get_state()
    #     rng2.set_state(state)
    # else:  # 'cuda'
    #     state = rng.graphsafe_get_state()
    #     rng2.graphsafe_set_state(state)
    # NOTE: may no longer be maintained, but still works in torch 2.9.*

    state = rng.get_offset()
    rng2.set_offset(state)

    check = []

    for _ in range(n_trials):
        A = torch.normal(
            mean=0,
            std=1,
            size=sample_shape,
            generator=rng,
            device=torch_device,
        )
        B = torch.normal(
            mean=0,
            std=1,
            size=sample_shape,
            generator=rng2,
            device=torch_device,
        )

        check.append(torch.equal(A, B))

    assert all(check), "GPU RNG restart test failed."
