r"""Testing numpy and torch warmstart options for random number generators in a
MPI multi-GPU setting. States are saved to and loaded from a ``.h5`` file."""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)

# NOTE:
# ! It seems temporary pytest path ``tmp_path`` does not work with MPI (test
# ! hanging forever). For now, create a file at the root of the project, and
# ! delete it after the test passes

import os

import h5py
import numpy as np
import pytest
import torch
from mcmc.DataManager.warmstart_rng_mpi import (
    load_rng_np_mpi,
    load_rng_offset_torch_mpi,
    load_rng_torch_mpi,
    save_rng_np_mpi,
    save_rng_offset_torch_mpi,
    save_rng_torch_mpi,
)
from mpi4py import MPI

pytestmark = pytest.mark.mpi


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


@pytest.fixture
def device(comm):
    # reference use
    # https://pytorch.org/docs/stable/generated/torch.set_default_device.html#torch-set-default-device
    # https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
    # ! see how to keep contiguous processes acting on the same GPU. This may
    # ! have an impact on the simulations of Maxime
    d = (
        torch.device("cuda", comm.Get_rank() % torch.cuda.device_count())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    torch.cuda.set_device(d)
    return d


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
def test_warmstart_rng_np_mpi(comm, seed, new_seed, n_samples):
    r"""Test warmstart of a numpy random number generator by restoring its
    state in a distributed setting."""
    filename = "warmstart_rng_numpy_mpi.h5"

    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        ss = np.random.SeedSequence(seed)
        # spawn off nworkers child SeedSequences to pass to child processes.
        child_seed = np.array(ss.spawn(size))
    else:
        child_seed = None

    local_seed = comm.scatter(child_seed, root=0)
    local_rng = np.random.default_rng(local_seed)

    x = local_rng.standard_normal(size=(n_samples,))
    assert np.linalg.norm(x) > 0

    with h5py.File(filename, "w", driver="mpio", comm=comm) as f:
        save_rng_np_mpi(rank, size, local_rng, f)

    y = local_rng.standard_normal(size=(n_samples,))

    new_local_rng = np.random.default_rng(new_seed)
    with h5py.File(filename, "r") as f:
        load_rng_np_mpi(rank, new_local_rng, f)

    z = new_local_rng.standard_normal(size=(n_samples,))

    # check y = z on each process
    local_consistency_check = np.array([np.allclose(y, z)])
    global_consistency_check = np.array([False])
    assert local_consistency_check[0]

    # reduce "local_consistency_check" on the root
    comm.Reduce(
        [local_consistency_check, MPI.C_BOOL],
        [global_consistency_check, MPI.C_BOOL],
        op=MPI.LAND,
        root=0,
    )

    comm.Barrier()
    if rank == 0:
        assert global_consistency_check
        os.remove(filename)
    pass


@pytest.mark.torch
def test_warmstart_rng_offset_torch_mpi(comm, seed, new_seed, n_samples, device):
    r"""Test warmstart of a torch random number generator using the offset from
    an initial seed. Tested in a distributed setting."""
    filename = "warmstart_rng_offset_torch_mpi.h5"
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ! doubt about the statistical robustness of multi-GPU sampling with
    # ! torch, compared to cupy/numpy, where this feature is explicitly
    # ! documented (Philox, ...)
    # https://docs.cupy.dev/en/stable/reference/random.html
    # https://numpy.org/doc/stable/reference/random/parallel.html
    # TODO: check https://pytorch.org/docs/stable/notes/randomness.html for
    # parallel random number generation with torch
    rng = torch.Generator(device=device).manual_seed(int("{}{}".format(rank, seed)))

    print("Worker: {}, GPU device: {}".format(rank, device))

    x = torch.randn((n_samples,), generator=rng, device=device)
    assert torch.linalg.vector_norm(x) > 0

    with h5py.File(filename, "w", driver="mpio", comm=comm) as f:
        save_rng_offset_torch_mpi(rank, size, rng, seed, f)

    y = torch.randn((n_samples,), generator=rng, device=device)

    new_rng = torch.Generator(device=device).manual_seed(
        int("{}{}".format(rank, new_seed))
    )
    with h5py.File(filename, "r", driver="mpio", comm=comm) as f:
        load_rng_offset_torch_mpi(rank, new_rng, f)

    z = torch.randn((n_samples,), generator=new_rng, device=device)

    assert torch.allclose(y, z)

    comm.Barrier()
    if rank == 0:
        os.remove(filename)
    pass


@pytest.mark.torch
def test_warmstart_rng_torch_mpi(comm, seed, new_seed, n_samples, device):
    r"""Test warmstart of a torch random number generator by restoring its
    state in a distributed setting."""
    filename = "warmstart_rng_torch_mpi.h5"
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ! doubt about the statistical robustness of multi-GPU sampling with
    # ! torch, compared to cupy/numpy, where this feature is explicitly
    # ! documented (Philox, ...)
    # https://docs.cupy.dev/en/stable/reference/random.html
    # https://numpy.org/doc/stable/reference/random/parallel.html
    # TODO: check https://pytorch.org/docs/stable/notes/randomness.html for
    # parallel random number generation with torch
    rng = torch.Generator(device=device).manual_seed(int("{}{}".format(rank, seed)))

    print("Worker: {}, GPU device: {}".format(rank, device))

    x = torch.randn((n_samples,), generator=rng, device=device)
    assert torch.linalg.vector_norm(x) > 0

    with h5py.File(filename, "w", driver="mpio", comm=comm) as f:
        save_rng_torch_mpi(rank, size, rng, seed, f)

    y = torch.randn((n_samples,), generator=rng, device=device)

    new_rng = torch.Generator(device=device).manual_seed(
        int("{}{}".format(rank, new_seed))
    )
    with h5py.File(filename, "r", driver="mpio", comm=comm) as f:
        load_rng_torch_mpi(rank, new_rng, f)

    z = torch.randn((n_samples,), generator=new_rng, device=device)

    assert torch.allclose(y, z)

    comm.Barrier()
    if rank == 0:
        os.remove(filename)
    pass


if __name__ == "__main__":
    seed = 1234
    new_seed = 2589
    n_samples = 1000
    comm = MPI.COMM_WORLD
    device = (
        torch.device("cuda", comm.Get_rank() % torch.cuda.device_count())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    torch.cuda.set_device(device)

    test_warmstart_rng_np_mpi(comm, seed, new_seed, n_samples)
    test_warmstart_rng_offset_torch_mpi(comm, seed, new_seed, n_samples, device)
    test_warmstart_rng_torch_mpi(comm, seed, new_seed, n_samples, device)
