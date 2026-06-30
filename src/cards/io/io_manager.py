r"""Object that handles any reading/writing on disk with parallel memory access."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.

# TODO: revise usefullness of all the methods included, possibly simplify implementation
# TODO: add more h5py save options in the interface (rdcc_nbytes, compression, compression_opts, chunk local_sizes, ...)
# NOTE: using h5py.read_direct / h5py.write_direct to avoid any copy of arrays

from typing import Optional

import h5py
import numpy as np
import torch

from cards.io.warmstart_rng import (
    array_to_int,
    int_to_array,
    load_rng_np,
    load_rng_offset_torch,
    save_rng_np,
)
from cards.io.warmstart_rng_mpi import load_rng_np_mpi, save_rng_np_mpi


class IOManager:
    """Utility class to handle I/O writing to disk in HDF5 files, either in serial of
    distributed mode.

    Parameters
    ----------
    ckpt_size : int
        Number of samples to collect before flushing results to disk.
    local_sizes : Optional[dict], optional
        In distributed mode, encodes the shape of the local tiles to be
        saved to disk, by default None.
    global_sizes : Optional[dict], optional
        In distributed mode, dictionary containing the shape of the global variables collectively saved to disk. By default None.
    local_slices : Optional[dict], optional
        In distributed mode, dictionary of slicer objects to extract, for eahc variable of interest, the local tile to be saved to disk from
        the local facet avaiable (i.e., buffer representing an array tile
        with the associated ghost-cell). By default None.

    Attributes
    ----------
    ckpt_size: int
        Number of samples to collect before flushing results to disk.

    global_sizes : dict, optional
        In distributed mode, dictionary containing the shape of the global variables collectively saved to disk. By default None.
    local_slices : dict, optional
        In distributed mode, dictionary of slicer objects to extract, for eahc variable of interest, the local tile to be saved to disk from
        the local facet avaiable (i.e., buffer representing an array tile
        with the associated ghost-cell). By default None.
    """

    def __init__(
        self,
        ckpt_size: int | None = None,
        local_sizes: dict | None = None,
        global_sizes: dict | None = None,
        local_slices: dict | None = None,
    ) -> None:
        self.ckpt_size = ckpt_size
        self.local_sizes = local_sizes
        self.global_sizes = global_sizes
        self.local_slices = local_slices

        # if self.local_sizes:
        #     for key in self.local_sizes:
        #         self.local_sizes[key] = tuple((ckpt_size, *self.local_sizes[key]))

        # if self.global_sizes:
        #     for key in self.global_sizes:
        #         self.global_sizes[key] = tuple((ckpt_size, *self.global_sizes[key]))

        # if self.local_slices:
        #     for key in self.local_slices:
        #         # add one axis for the batch dimension
        #         self.local_slices[key] = np.s_[slice(None), *self.local_slices[key]]

    def save_dict(
        self,
        data: dict,
        file: h5py.File,
        global_sizes: Optional[dict] = None,
        slices: Optional[dict] = None,
    ) -> None:
        """Save the content of an input dictionary in an .h5 file.

        Parameters
        ----------
        data : dict
            Dictionary containing data to write on disk.
        file : h5py.File
            File on which the data will be written.
        global_size:
            Dictionary containing the global dimensions of the buffers.
        slices: dict
            Dictionary containing the indexes of the vertices delimiting the position of the local buffer in the global buffer.

        Caution
        -------
        In distributed settings, the ``h5py.File`` object passed as input is
        expected to correspond to a file opened in parallel mode (i.e., with
        the ``h5py`` flag ``driver="mpio"``).
        """
        for key in data:
            print(key, global_sizes, slices)
            var = key.split("_")[0]
            if global_sizes is None:
                buffer_size = data[key].shape
            else:
                buffer_size = global_sizes[var]

            if slices is None:
                local_slice = slice(None)
            else:
                local_slice = slices[var]
            dset = file.create_dataset(
                name=key, shape=buffer_size, dtype=data[key].dtype
            )
            dset.write_direct(data[key], dest_sel=local_slice)

    def save_array(
        self,
        data: np.ndarray,
        file: h5py.File,
        name: str,
        global_size: Optional[np.ndarray] = None,
        local_slice: Optional[slice] = slice(None),
    ) -> None:
        """Save an input array to a specific ``.h5`` file.

        Parameters
        ----------
        data : np.ndarray
            Input array to write on file.
        file : h5py.File
            File on which the data will be written.
        name : str
            Name of the data-field in the file.
        global_size:
            In distribuyted mode, global shape of the buffer in which the local
            input array will be stored. By default None.
        local_slice: slice
            Slicer object delimiting the position of the local array tile
            within the global array field in the ``.h5`` file.
        """
        if global_size is None:
            global_size = data.shape
        dset = file.create_dataset(name, global_size, dtype=data.dtype)
        dset.write_direct(data, dest_sel=local_slice)

    def save_seed(self, seed: int, rank: int, comm_size: int, file: h5py.File) -> None:
        """In distributed mode, save the random generator seed used on each process.

        Parameters
        ----------
        seed : int
            Local seed.
        rank : int
            Rank of the process.
        comm_size : int
            Number of process.
        file : h5py.File
            File to be written on.

        Caution
        -------
        In distributed settings, the ``h5py.File`` object passed as input is
        expected to correspond to a file opened in parallel mode (i.e., with
        the ``h5py`` flag ``driver="mpio"``).
        """
        dset = file.create_dataset("seed", (comm_size,), dtype=int)
        dset[rank] = seed

    def save_local_array(self, data: np.ndarray, name: str, file: h5py.File) -> None:
        """Save an array on an .h5 file.

        Parameters
        ----------
        data : np.ndarray
            Local array.
        name : str
            Name of the variable.
        file : h5py.File
            File to be written on.

        Caution
        -------
        The ``h5py.File`` object passed as input is expected to correspond to a
        file opened in serial mode.
        """
        dset = file.create_dataset(name, data.shape, dtype=data.dtype)
        dset.write_direct(data)
        return

    def save_thread_array(
        self, data: np.ndarray, rank: int, comm_size: int, name: str, file: h5py.File
    ) -> None:
        """Simultaneously save an array along each thread.

        Parameters
        ----------
        data : np.ndarray
            Local array.
        rank : int
            Rank of the current thread.
        comm_size : int
            Number of thread available int he commuicator.
        name : str
            Name of the datafield.
        file : h5py.File
            File where to writte the data.
        """
        dset = file.create_dataset(name, (comm_size, *data.shape), dtype=data.dtype)
        dset[rank, ...] = data
        return

    def save_thread_scalar(
        self, data: float, rank: int, comm_size: int, name: str, file: h5py.File
    ) -> None:
        """Simultaneously save a scalar along each thread.

        Parameters
        ----------
        data : np.ndarray
            Local scalar.
        rank : int
            Rank of the current thread.
        comm_size : int
            Number of thread available int he commuicator.
        name : str
            Name of the datafield.
        file : h5py.File
            File where to writte the data.
        """
        dset = file.create_dataset(name, comm_size, dtype=data.dtype)
        dset[rank] = data
        return

    def load_h5(
        self,
        file: h5py.File,
        local_sizes: Optional[dict] = None,
        slices: Optional[dict] = None,
    ) -> dict:
        """Load a .h5 file and load the local value of the array on each process.
        It expects the given file to be open in parallel mode.

        Parameters
        ----------
        file : h5py.File
            File to be read.
        local_sizes : dict
            Dictionary containing the size of each local buffers.
        slices : dict
            Dictionary containing the slices of each variables, different on each thread.

        Returns
        -------
        dict
            Dictonary containing the local value of each variable.
        """
        data = {}
        if slices is None:
            for key in file.keys():
                if file[key].size > 1:
                    data[key] = file[key][:]
                else:
                    data[key] = file[key][()]
        else:
            for key in slices.keys():
                data[key] = np.zeros(local_sizes[key])
                file[key].read_direct(data[key], source_sel=slices[key])
        return data

    def save_rng(
        self,
        rng: np.random.Generator,
        file: h5py.File,
        rank: int = 0,
        comm_size: int = 0,
    ) -> None:
        """Save the internal state of all the random number generators used
        on each MPI process.

        Parameters
        ----------
        comm : MPI.Comm
            Current MPI communicator.
        rng : np.random.Generator
            Local random number generator.
        file : h5py.File
            File to be written on.
        """
        if comm_size == 0:
            save_rng_np(rng, file)
        else:
            save_rng_np_mpi(rank, comm_size, rng, file)
        return

    def load_rng(
        self, rng: np.random.Generator, file: h5py.File, rank: Optional[int] = None
    ) -> None:
        """Load the internal state of the random number generator for each process.

        Parameters
        ----------
        rng : np.random.Generator
            Local random number generator.
        file : h5py.File
            File to be read.
        rank : int
            Rank of the process.
        """
        if rank is None:
            load_rng_np(rng, file)
        else:
            load_rng_np_mpi(rank, rng, file)
        return

    def save_rng_torch(
        self,
        rng: torch._C.Generator,
        seed: int,
        h5file: h5py.File,
        gpu_id: Optional[int] = 0,
        nb_gpu: int = 1,
    ) -> None:
        r"""Save current state of a pytorch random number generator in a .h5 file
        using the offset from the initial seed state.

        Parameters
        ----------
        rng : torch._C.Generator
            Pytorch random number generator on the GPU.
        seed : int
            Seed used to initialize the generator.
        h5file : h5py.File
            Handle to a `.h5` file to save the state of the generator.
        gpu_id : int
            Identifaint of the current device.
        comm_size : int
            Number of GPU available.
        Note
        ----
        Requires ``pythorch>=2.5``. Only supported for generators on the GPU.
        """
        seed_array = int_to_array(seed)
        dset_seed = h5file.create_dataset(
            "seed", (nb_gpu, *seed_array.shape), dtype=seed_array.dtype
        )
        dset_seed[gpu_id, ...] = seed_array

        offset_array = int_to_array(rng.get_offset())
        dset_offset = h5file.create_dataset(
            "offset",
            (nb_gpu, *offset_array.shape),
            dtype=offset_array.dtype,
        )
        dset_offset[gpu_id, ...] = offset_array
        return

    def load_rng_torch(
        self, rng: torch.Generator, h5file: h5py.File, gpu_id: Optional[int] = None
    ):
        r"""Load the state of several pytorch random number generators from a .h5 file using the offset from an initial seed state.

        Parameters
        ----------
        rng : torch._C.Generator
            Pytorch local random number generator.
        h5file : h5py.File
            Handle to a `.h5` file to save the state of the generator.
        gpu_id : int
            Identifiant of the current device.

        Note
        ----
        Requires ``pythorch>=2.5``. Only supported for generators on the GPU.
        """
        # ! an offset is relative to some initial seed, which also needs to be
        # ! loaded and set
        if gpu_id is None:
            load_rng_offset_torch(rng, h5file)
        else:
            seed = array_to_int(h5file["seed"][gpu_id, ...])
            rng.manual_seed(seed)
            offset = array_to_int(h5file["offset"][gpu_id, ...])
            rng.set_offset(offset)
        return

    def load_states(self, file: h5py.File, vars: list[str]) -> dict:
        r"""Load the state of the variables to be sampled from a .h5 file.

        Parameters
        ----------
        file : h5py.File
            Handle to a `.h5` file to load the state of the variables.
        vars : list[str]
            List of variable names to be loaded.

        Returns
        -------
        dict
            Dictionary containing the state of the variables.
        """
        states = {}
        for var in vars:
            states[var] = file[var][:]
        return states
