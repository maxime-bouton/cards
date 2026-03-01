r"""Object that handles any reading/writing on disk with parallel memory access."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.

# TODO: revise usefullness of all the methods included, possibly simplify implementation
# TODO: add more h5py save options in the interface (rdcc_nbytes, compression, compression_opts, chunk sizes, ...)
# NOTE: using h5py.read_direct / h5py.write_direct to avoid any copy of arrays

from typing import Optional

import h5py
import numpy as np
import torch

from cards.backend import xp
from cards.data_manager.warmstart_rng import (
    array_to_int,
    int_to_array,
    load_rng_np,
    load_rng_offset_torch,
    save_rng_np,
)
from cards.data_manager.warmstart_rng_mpi import load_rng_np_mpi, save_rng_np_mpi


class DataManager:
    """Utility class to handle I/O writing to disk in HDF5 files, either in serial of distributed mode.

    Attributes
    ----------
    _save_full_batch : bool
        Save mode whereby a batch of several consecutive samples shoud be
        saved to disk, by default False.
    full_batch :
        Dictionary contaning, for each variable of interest, the batch of
        samples to be saved to disk.
    global_sizes : dict, optional
            In distributed mode, dictionary containing the shape of the global variables collectively saved to disk. By default None.
    local_slices : dict, optional
        In distributed mode, dictionary of slicer objects to extract, for eahc variable of interest, the local tile to be saved to disk from
        the local facet avaiable (i.e., buffer representing an array tile
        with the associated ghost-cell). By default None.
    """

    def __init__(
        self,
        batch_size: int = 1,
        save_full_batch: bool = False,
        sizes: Optional[dict] = None,
        global_sizes: Optional[dict] = None,
        local_slices: Optional[dict] = None,
    ) -> None:
        """DataManager constructor.

        Parameters
        ----------
        batch_size : int, optional
            Number of samples to collect before flushing results to disk, by
            default 1.
        save_full_batch : bool, optional
            Save mode whereby a batch of several consecutive samples shoud be
            saved to disk, by default False.
        sizes : Optional[dict], optional
            In distributed mode, encodes the shape of the local tiles to be
            saved to disk, by default None.
        global_sizes : Optional[dict], optional
            In distributed mode, dictionary containing the shape of the global variables collectively saved to disk. By default None.
        local_slices : Optional[dict], optional
            In distributed mode, dictionary of slicer objects to extract, for eahc variable of interest, the local tile to be saved to disk from
            the local facet avaiable (i.e., buffer representing an array tile
            with the associated ghost-cell). By default None.
        """
        self._save_full_batch = save_full_batch
        if save_full_batch:
            self.full_batch = {}
            self.global_sizes = {}
            self.local_slices = None

            if sizes is not None:
                for key in sizes:
                    self.full_batch[key] = xp.zeros((batch_size, *sizes[key]))
                    self.global_sizes[key] = tuple((batch_size, *sizes[key]))
            if global_sizes is not None:
                for key in global_sizes:
                    self.global_sizes[key] = tuple((batch_size, *global_sizes[key]))

            if local_slices is not None:
                self.local_slices = {}
                for key in local_slices:
                    # add one axis for the batch dimension
                    self.local_slices[key] = np.s_[slice(None), *local_slices[key]]

    def store_states(self, states, num_iter):
        """Store a new state to the batch of samples for all the variables to
        be saved to disk."""
        for key in self.full_batch:
            self.full_batch[key][num_iter, ...] = states[key]

    def save_batch(self, file: h5py.File, from_gpu: bool = False):
        """Save batch of samples to disk."""
        for key in self.full_batch:
            buffer_size = self.global_sizes[key]

            # TODO: see if this "if"-statement can be simplified or removed
            if self.local_slices is None:
                local_slice = slice(None)
            else:
                local_slice = self.local_slices[key]

            dset = file.create_dataset(
                name="batch/" + key, shape=buffer_size, dtype=self.full_batch[key].dtype
            )

            if not from_gpu:
                dset.write_direct(self.full_batch[key], dest_sel=local_slice)
            else:
                dset.write_direct(self.full_batch[key].get(), dest_sel=local_slice)

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
            if global_sizes is None:
                buffer_size = data[key].shape
            else:
                buffer_size = global_sizes[key]

            if slices is None:
                local_slice = slice(None)
            else:
                local_slice = slices[key]
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
