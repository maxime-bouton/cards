r"""Testing numpy and torch warmstart options for random number generators. States are saved to and loaded from a ``.h5`` file."""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)

from os.path import join

import h5py
import numpy as np
import pytest
import torch
from mcmc.DataManager.warmstart_rng import (
    load_rng_np,
    load_rng_offset_torch,
    load_rng_torch,
    save_rng_np,
    save_rng_offset_torch,
    save_rng_torch,
)


@pytest.fixture
def device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@pytest.fixture
def seed():
    return 1234


@pytest.fixture
def new_seed():
    return 1556


@pytest.fixture
def n_samples():
    return 1000


@pytest.mark.numpy
def test_warmstart_rng_np(tmp_path, seed, new_seed, n_samples):
    r"""Test warmstart of a numpy random number generator by restoring its
    state."""
    if tmp_path is not None:
        tmp_path_str = tmp_path.as_posix()
    else:
        tmp_path_str = ""
    filename = join(tmp_path_str, "warmstart_numpy_rng.h5")
    rng = np.random.default_rng(seed)

    x = rng.standard_normal(size=(n_samples,))
    assert np.linalg.norm(x) > 0

    with h5py.File(filename, "w") as f:
        save_rng_np(rng, f)

    y = rng.standard_normal(size=(n_samples,))

    new_rng = np.random.default_rng(new_seed)
    with h5py.File(filename, "r") as f:
        load_rng_np(new_rng, f)

    z = new_rng.standard_normal(size=(n_samples,))

    assert np.allclose(y, z)


@pytest.mark.torch
def test_warmstart_rng_offset_torch(tmp_path, seed, new_seed, n_samples, device):
    r"""Test warmstart of a torch random number generator using the offset from
    an initial seed.
    """
    if tmp_path is not None:
        tmp_path_str = tmp_path.as_posix()
    else:
        tmp_path_str = ""
    filename = join(tmp_path_str, "warmstart_torch_rng.h5")

    rng = torch.Generator(device=device).manual_seed(seed)

    x = torch.randn((n_samples,), generator=rng, device=device)
    assert torch.linalg.vector_norm(x) > 0

    with h5py.File(filename, "w") as f:
        save_rng_offset_torch(rng, seed, f)

    y = torch.randn((n_samples,), generator=rng, device=device)

    new_rng = torch.Generator(device=device).manual_seed(new_seed)
    with h5py.File(filename, "r") as f:
        load_rng_offset_torch(new_rng, f)

    z = torch.randn((n_samples,), generator=new_rng, device=device)

    assert torch.allclose(y, z)


@pytest.mark.torch
def test_warmstart_rng_torch(tmp_path, seed, new_seed, n_samples, device):
    r"""Test warmstart of a torch random number generator by restoring its
    state."""
    if tmp_path is not None:
        tmp_path_str = tmp_path.as_posix()
    else:
        tmp_path_str = ""
    filename = join(tmp_path_str, "warmstart_torch_rng.h5")

    rng = torch.Generator(device=device).manual_seed(seed)

    x = torch.randn((n_samples,), generator=rng, device=device)
    assert torch.linalg.vector_norm(x) > 0

    with h5py.File(filename, "w") as f:
        save_rng_torch(rng, f)

    y = torch.randn((n_samples,), generator=rng, device=device)

    new_rng = torch.Generator(device=device).manual_seed(new_seed)
    with h5py.File(filename, "r") as f:
        load_rng_torch(new_rng, f)

    z = torch.randn((n_samples,), generator=new_rng, device=device)

    assert torch.allclose(y, z)


if __name__ == "__main__":
    seed = 1234
    new_seed = 2589
    n_samples = 1000
    tmp_path = None
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    test_warmstart_rng_np(tmp_path, seed, new_seed, n_samples)
    test_warmstart_rng_offset_torch(tmp_path, seed, new_seed, n_samples, device)
    test_warmstart_rng_torch(tmp_path, seed, new_seed, n_samples, device)
