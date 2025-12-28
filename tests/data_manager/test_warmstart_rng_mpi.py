r"""Testing numpy and torch warmstart options for random number generators in a
MPI multi-GPU setting. States are saved to and loaded from a ``.h5`` file."""

# NOTE:
# ! It seems temporary pytest path ``tmp_path`` does not work with MPI (test
# ! hanging forever). For now, create a file at the root of the project, and
# ! delete it after the test passes

import os

import h5py
import numpy as np
import pytest
import torch
from mpi4py import MPI

from cards.data_manager.warmstart_rng_mpi import (
    load_rng_np_mpi,
    load_rng_offset_torch_mpi,
    load_rng_torch_mpi,
    save_rng_np_mpi,
    save_rng_offset_torch_mpi,
    save_rng_torch_mpi,
)


@pytest.fixture
def sample_shape():
    return (1000,)


@pytest.mark.mpi
@pytest.mark.cpu
def test_warmstart_rng_np_mpi(tmp_path, comm, seed, seed2, sample_shape):
    r"""Test warmstart of a numpy random number generator by restoring its
    state in a distributed setting."""
    rank = comm.Get_rank()
    filename = ""
    if rank == 0:
        if tmp_path is not None:
            tmp_path_str = tmp_path.as_posix()
        else:
            tmp_path_str = ""
        filename = os.path.join(tmp_path_str, "warmstart_rng_numpy_mpi.h5")

    filename = comm.bcast(filename, 0)

    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        ss = np.random.SeedSequence(seed)
        # spawn off nworkers child SeedSequences to pass to child processes.
        child_seed = ss.spawn(size)
    else:
        child_seed = None

    local_seed = comm.scatter(child_seed, root=0)
    local_rng = np.random.default_rng(local_seed)

    x = local_rng.standard_normal(size=sample_shape)
    assert np.linalg.norm(x) > 0

    with h5py.File(filename, "w", driver="mpio", comm=comm) as f:
        save_rng_np_mpi(rank, size, local_rng, f)

    y = local_rng.standard_normal(size=sample_shape)

    new_local_rng = np.random.default_rng(seed2)
    with h5py.File(filename, "r") as f:
        load_rng_np_mpi(rank, new_local_rng, f)

    z = new_local_rng.standard_normal(size=sample_shape)

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


@pytest.mark.mpi
@pytest.mark.gpu
def test_warmstart_rng_offset_torch_mpi(
    tmp_path, comm, seed, seed2, sample_shape, torch_device
):
    r"""Test warmstart of a torch random number generator using the offset from
    an initial seed. Tested in a distributed setting."""
    filename = "warmstart_rng_offset_torch_mpi.h5"
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        if tmp_path is not None:
            tmp_path_str = tmp_path.as_posix()
        else:
            tmp_path_str = ""
        filename = os.path.join(tmp_path_str, "warmstart_rng_offset_torch_mpi.h5")

    filename = comm.bcast(filename, 0)

    # ! doubt about the statistical robustness of multi-GPU sampling with
    # ! torch, compared to cupy/numpy, where this feature is explicitly
    # ! documented (Philox, ...)
    # https://docs.cupy.dev/en/stable/reference/random.html
    # https://numpy.org/doc/stable/reference/random/parallel.html
    # TODO: check https://pytorch.org/docs/stable/notes/randomness.html for
    # parallel random number generation with torch
    rng = torch.Generator(device=torch_device).manual_seed(
        int("{}{}".format(rank, seed))
    )

    print("Worker: {}, GPU device: {}".format(rank, torch_device))

    x = torch.randn(sample_shape, generator=rng, device=torch_device)
    assert torch.linalg.vector_norm(x) > 0

    with h5py.File(filename, "w", driver="mpio", comm=comm) as f:
        save_rng_offset_torch_mpi(rank, size, rng, seed, f)

    y = torch.randn(sample_shape, generator=rng, device=torch_device)

    new_rng = torch.Generator(device=torch_device).manual_seed(
        int("{}{}".format(rank, seed2))
    )
    with h5py.File(filename, "r", driver="mpio", comm=comm) as f:
        load_rng_offset_torch_mpi(rank, new_rng, f)

    z = torch.randn(sample_shape, generator=new_rng, device=torch_device)

    assert torch.allclose(y, z)

    comm.Barrier()
    if rank == 0:
        os.remove(filename)
    pass


@pytest.mark.mpi
@pytest.mark.gpu
def test_warmstart_rng_torch_mpi(
    tmp_path, comm, seed, seed2, sample_shape, torch_device
):
    r"""Test warmstart of a torch random number generator by restoring its
    state in a distributed setting."""
    filename = "warmstart_rng_torch_mpi.h5"
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        if tmp_path is not None:
            tmp_path_str = tmp_path.as_posix()
        else:
            tmp_path_str = ""
        filename = os.path.join(tmp_path_str, "warmstart_rng_torch_mpi.h5")

    filename = comm.bcast(filename, 0)

    # ! doubt about the statistical robustness of multi-GPU sampling with
    # ! torch, compared to cupy/numpy, where this feature is explicitly
    # ! documented (Philox, ...)
    # https://docs.cupy.dev/en/stable/reference/random.html
    # https://numpy.org/doc/stable/reference/random/parallel.html
    # TODO: check https://pytorch.org/docs/stable/notes/randomness.html for
    # parallel random number generation with torch
    rng = torch.Generator(device=torch_device).manual_seed(
        int("{}{}".format(rank, seed))
    )

    print("Worker: {}, GPU device: {}".format(rank, torch_device))

    x = torch.randn(sample_shape, generator=rng, device=torch_device)
    assert torch.linalg.vector_norm(x) > 0

    with h5py.File(filename, "w", driver="mpio", comm=comm) as f:
        save_rng_torch_mpi(rank, size, rng, seed, f)

    y = torch.randn(sample_shape, generator=rng, device=torch_device)

    new_rng = torch.Generator(device=torch_device).manual_seed(
        int("{}{}".format(rank, seed2))
    )
    with h5py.File(filename, "r", driver="mpio", comm=comm) as f:
        load_rng_torch_mpi(rank, new_rng, f)

    z = torch.randn(sample_shape, generator=new_rng, device=torch_device)

    assert torch.allclose(y, z)

    comm.Barrier()
    if rank == 0:
        os.remove(filename)
    pass
