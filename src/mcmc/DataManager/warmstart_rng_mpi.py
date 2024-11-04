r"""Save state of random number generators in numpy
(``np.random.Generator`` class) and torch (``torch._C.Generator`` class) to a
``.h5`` with a MPI setting.
"""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)

from os.path import join

import h5py
import numpy as np
import torch
from mpi4py import MPI

from mcmc.DataManager.warmstart_rng import (
    array_to_int,
    extract_rng_state,
    int_to_array,
    restore_rng_state,
)


def save_rng_np_mpi(
    comm: MPI.Comm, rng: np.random.Generator, h5file: h5py.File
) -> None:
    r"""Save current state of numpy random number generators in a .h5 file in
    an MPI application.

    Parameters
    ----------
    comm : MPI.Comm
        Current MPI communicator.
    rng : np.random.Generator
        Numpy random number generator.
    h5file : h5py.File
        Handle to a `.h5` file to save the state of the generator on the
        current process. The file should be opened collectively with "mpio"
        driver.

    Note
    ----
    Requires ``numpy>=2.0``.
    """
    rank = comm.Get_rank()

    state_array, inc_array = extract_rng_state(rng)
    dset_state = []
    dset_inc = []
    for r in range(comm.Get_size()):
        dset_state.append(
            h5file.create_dataset(
                join(str(r), "rng_state_array"), state_array.shape, dtype=np.uint8
            )
        )
        dset_inc.append(
            h5file.create_dataset(
                join(str(r), "rng_inc_array"), inc_array.shape, dtype=np.uint8
            )
        )
    dset_state[rank][:] = state_array
    dset_inc[rank][:] = inc_array
    return


def load_rng_np_mpi(
    comm: MPI.Comm, rng: np.random.Generator, h5file: h5py.File
) -> None:
    r"""Load the state of a numpy random number generator from a .h5 file in
    an MPI application.

    Parameters
    ----------
    comm : MPI.Comm
        Current MPI communicator.
    rng : np.random.Generator
        Numpy random number generator.
    h5file : h5py.File
        Handle to a `.h5` file to save the state of the generator on the
        current process. The file should be opened collectively with "mpio"
        driver.

    Note
    ----
    Requires ``numpy>=2.0``.
    """
    rank = comm.Get_rank()
    restore_rng_state(
        rng,
        h5file[join(str(rank), "rng_state_array")][:],
        h5file[join(str(rank), "rng_inc_array")][:],
    )
    return


def save_rng_offset_torch_mpi(
    comm: MPI.Comm, rng: torch._C.Generator, seed: int, h5file: h5py.File
):
    """Save current state of a pytorch random number generator in a .h5 file
    within an MPI application. Uses the offset from the initial seed state.

    Parameters
    ----------
    comm : MPI.Comm
        Current MPI communicator.
    rng : torch._C.Generator
        Pytorch random number generator.
    seed : int
        Seed used to initialize the generator.
    h5file : h5py.File
        Handle to a `.h5` file to save the state of the generator on the
        current process. The file should be opened collectively with "mpio"
        driver.

    Note
    ----
    Requires ``pythorch>=2.5``.
    """
    # ! 1. built-in int type from Python can be very large, and cannot be saved
    # as is to .h5. They need to be converted to hex format (later to an array of ints) to be saved in an .h5 file
    # ! 2. the offset is relative to the initial seed, and thus needs to be saved
    rank = comm.Get_rank()
    seed_array = int_to_array(seed)
    offset_array = int_to_array(rng.get_offset())

    dset_seed = []
    dset_offset = []
    for r in range(comm.Get_size()):
        dset_seed.append(
            h5file.create_dataset(
                join(str(r), "seed"), seed_array.shape, dtype=np.uint8
            )
        )
        dset_offset.append(
            h5file.create_dataset(
                join(str(r), "offset"), offset_array.shape, dtype=np.uint8
            )
        )
    dset_seed[rank][:] = seed_array
    dset_offset[rank][:] = offset_array
    return


def load_rng_offset_torch_mpi(
    comm: MPI.Comm, rng: torch._C.Generator, h5file: h5py.File
):
    r"""Load the state of a pytorch random number generator from a .h5 file
    within an MPI application. Uses the offset from the initial seed state.

    Parameters
    ----------
    comm : MPI.Comm
        Current MPI communicator.
    rng : torch._C.Generator
        Pytorch random number generator.
    h5file : h5py.File
        Handle to a `.h5` file to save the state of the generator on the
        current process. The file should be opened collectively with "mpio"
        driver.

    Note
    ----
    Requires ``pythorch>=2.5``.
    """
    # ! an offset is relative to some initial seed, which also needs to be
    # ! loaded and set
    rank = comm.Get_rank()
    seed = array_to_int(h5file[join(str(rank), "seed")][:])
    rng.manual_seed(int("{}{}".format(rank, seed)))
    offset = array_to_int(h5file[join(str(rank), "offset")][:])
    rng.set_offset(offset)

    return


def save_rng_torch_mpi(
    comm: MPI.Comm, rng: torch._C.Generator, seed: int, h5file: h5py.File
):
    """Save current state of a pytorch random number generator in a .h5 file within an MPI application.

    Parameters
    ----------
    comm : MPI.Comm
        Current MPI communicator.
    rng : torch._C.Generator
        Pytorch random number generator.
    h5file : h5py.File
        Handle to a `.h5` file to save the state of the generator on the
        current process. The file should be opened collectively with "mpio"
        driver.
    """
    rank = comm.Get_rank()
    current_state = rng.get_state()

    dset = []
    for r in range(comm.Get_size()):
        dset.append(
            h5file.create_dataset(
                join(str(r), "torch_rng_state"),
                current_state.shape,
                dtype=str(current_state.dtype).split(".")[-1],
            )
        )
    dset[rank][:] = current_state
    return


def load_rng_torch_mpi(comm: MPI.Comm, rng: torch._C.Generator, h5file: h5py.File):
    r"""Load the state of a pytorch random number generator from a .h5 file
    within an MPI application.

    Parameters
    ----------
    comm : MPI.Comm
        Current MPI communicator.
    rng : torch._C.Generator
        Pytorch random number generator.
    h5file : h5py.File
        Handle to a `.h5` file to save the state of the generator on the
        current process. The file should be opened collectively with "mpio"
        driver.
    """
    rank = comm.Get_rank()

    # ! the loaded state should a priori be move to the GPU device, but at the
    # ! moment.
    # ! rng.set_state yields an error (state should be a torch.ByteTensor)
    # ! for the moment, loaded state is kept on the cpu
    loaded_state = torch.tensor(
        h5file[join(str(rank), "torch_rng_state")][:], dtype=torch.uint8, device="cpu"
    )
    rng.set_state(loaded_state)

    return
