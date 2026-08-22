r"""Object that handles any reading/writing on disk with parallel memory access."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

# TODO: add more h5py save options in the interface
# (rdcc_nbytes, compression, compression_opts, chunk local_sizes, ...)

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import torch

import cards.backend as xp
from cards.core.execution_context import ExecutionContext
from cards.random import restore_rng, serialize_rng


class IOManager:
    """Utility class to handle I/O writing and reading in HDF5 files.

    It operates seamlessly in either serial or distributed (MPI) modes,
    leveraging the provided execution context.

    Parameters
    ----------
    ctx : ExecutionContext
        The execution context defining if the run is serial or MPI, and
        storing the communicator and rank information.
    """

    def __init__(self, ctx: ExecutionContext) -> None:
        self.ctx = ctx

    @contextmanager
    def open(
        self,
        path: str | Path,
        mode: str = "r",
        force_serial: bool = False,
    ) -> Generator[h5py.File, None, None]:
        """Context manager to open an HDF5 file across all ranks.

        Automatically configures the appropriate ``h5py`` backend (standard serial
        or parallel MPIO) based on the current execution context.

        Parameters
        ----------
        path : str | pathlib.Path
            Path to the HDF5 file to open.
        mode : str, optional
            File access mode (e.g., 'r', 'w', 'r+', 'a', 'x'). Default is 'r'.
        force_serial : bool, optional
            If True, forces the file to be opened in standard serial mode on every
            rank, bypassing the parallel MPIO driver even if an MPI context is active.
            Default is False.

        Yields
        ------
        h5py.File
            The opened HDF5 file object configured for the current context.

        Notes
        -----
        Use this method for collective array operations or when reading/writing
        file-per-rank datasets. If ``force_serial=True`` is used during an MPI
        execution on a single shared file, the caller must ensure that only a
        single rank performs writes to avoid race conditions and file corruption.
        """
        if not self.ctx.is_mpi or force_serial:
            with h5py.File(path, mode) as f:
                yield f
        else:
            with h5py.File(path, mode, driver="mpio", comm=self.ctx.comm) as f:
                yield f

    @contextmanager
    def open_master_only(
        self,
        path: str | Path,
        mode: str = "r",
    ) -> Generator[h5py.File | None, None, None]:
        """Context manager to exclusively open an HDF5 file on the master rank.

        Uses the standard serial driver to open the file only on the master rank.
        All other ranks yield ``None`` without touching the file system.

        Parameters
        ----------
        path : str | pathlib.Path
            Path to the HDF5 file to open.
        mode : str, optional
            File access mode (e.g., 'r', 'w', 'r+', 'a', 'x'). Default is 'r'.

        Yields
        ------
        h5py.File | None
            The opened HDF5 file object on the master rank, and None on all other ranks.

        Notes
        -----
        This method is ideal for writing small, rank-independent global metadata (e.g.,
        configuration scalars, strings, or booleans as attributes). It avoids the
        overhead of collective MPI I/O and prevents file contention from workers.
        Operations inside the ``with`` block should handle the ``None`` case on worker
        ranks (e.g., ``if f is not None:``).
        """
        if not self.ctx.is_master:
            yield None
            return

        with h5py.File(path, mode) as f:
            yield f

    def write_config(
        self,
        file: h5py.File,
        data_dict: dict,
        group: str | None = None,
    ) -> None:
        """Write scalar, boolean, or string configuration values as HDF5 attributes.

        Attributes are used instead of datasets because configuration values are
        typically small and fixed-size, requiring no chunking or collective I/O.
        This method is designed to be called exclusively on the master rank,
        typically within a block opened via `open_master_only`.

        Parameters
        ----------
        file : h5py.File
            The HDF5 file object where the configuration will be written.
        data_dict : dict
            A dictionary containing the configuration keys and their corresponding
            values (scalars, booleans, or strings).
        group : str, optional
            The name of the HDF5 group under which to store the attributes. If
            empty or None, attributes are written to the root of the file.
            Default is None.
        """
        grp = file.require_group(group) if group else file
        for key, value in data_dict.items():
            grp.attrs[key] = value

    def read_config(
        self,
        file: h5py.File,
        group: str | None = None,
        keys: list[str] | None = None,
    ) -> dict:
        """Read configuration attributes from an HDF5 file.

        Parameters
        ----------
        file : h5py.File
            The HDF5 file object to read from.
        group : str, optional
            The name of the HDF5 group containing the configuration attributes.
            If empty or None, attributes are read from the root of the file.
            Default is None.
        keys : list[str], optional
            A specific list of configuration keys to retrieve. If None, all
            attributes present in the specified group are read. Default is None.

        Returns
        -------
        dict
            A dictionary containing the requested configuration keys and their
            corresponding values.

        Notes
        -----
        Unlike writing, reading configurations is safe to call from any rank,
        provided the file is opened appropriately for reading across all ranks.
        """
        grp = file[group] if group else file
        if keys is None:
            return dict(grp.attrs)
        return {key: grp.attrs[key] for key in keys}

    def write_array(
        self,
        file: h5py.File,
        name: str,
        data: xp.ndarray,
        global_shape: tuple | None = None,
        dest_slice: slice | tuple | None = None,
    ) -> None:
        """Write an input array to a specific dataset in an HDF5 file.

        Parameters
        ----------
        file : h5py.File
            File on which the data will be written.
        name : str
            Name of the data-field in the file.
        data : xp.ndarray
            Input array to write on file.
        global_shape : tuple, optional
            In distributed mode, the global shape of the buffer in which the local
            input array will be stored. If None, defaults to `data.shape`.
        dest_slice : slice or tuple, optional
            Slicer object delimiting the position of the local array tile
            within the global array field. If None, writes the entire array.

        Notes
        -----
        In MPI mode, if `dest_slice` is provided, this triggers a collective
        I/O operation. All ranks must call this method simultaneously.
        """
        # enforce C-contiguity for h5py's direct C-API calls
        # no overhead if the array is already contiguous
        # NOTE: breaks otherwise in `cpu` context
        if not data.flags.c_contiguous:
            data = xp.ascontiguousarray(data)

        if self.ctx.is_gpu:
            data = data.get()

        shape = global_shape if global_shape is not None else data.shape

        if name not in file:
            dset = file.create_dataset(name, shape=shape, dtype=data.dtype)
        else:
            dset = file[name]

        if dest_slice is None:
            dset[...] = data
        else:
            if self._is_collective(file, is_sliced=True):
                with dset.collective:
                    dset.write_direct(data, dest_sel=dest_slice)
            else:
                dset.write_direct(data, dest_sel=dest_slice)

    def read_array(
        self,
        file: h5py.File,
        name: str,
        source_slice: slice | tuple | None = None,
        out: xp.ndarray | None = None,
    ) -> xp.ndarray:
        """Read a single array or a chunk of it from an HDF5 file.

        Parameters
        ----------
        file : h5py.File
            File to be read.
        name : str
            Name of the variable to read.
        source_slice : slice or tuple, optional
            Slicer object delimiting the portion of the array to read.
        out : xp.ndarray, optional
            Pre-allocated local buffer to read the data into using `read_direct`.

        Returns
        -------
        xp.ndarray
            The read array or chunk.

        Notes
        -----
        In MPI mode, if `source_slice` is provided, this triggers a collective
        I/O operation. All ranks must call this method simultaneously.
        """
        dset = file[name]
        use_collective = self._is_collective(file, is_sliced=source_slice is not None)

        if out is not None:
            sel = source_slice if source_slice is not None else np.s_[...]
            host_out = np.empty(out.shape, dtype=out.dtype) if self.ctx.is_gpu else out

            if use_collective:
                with dset.collective:
                    dset.read_direct(host_out, source_sel=sel)
            else:
                dset.read_direct(host_out, source_sel=sel)

            if self.ctx.is_gpu:
                out.set(host_out)

            return out

        if source_slice is not None:
            if self._is_collective(file, source_slice is not None):
                with dset.collective:
                    res = dset[source_slice]
            else:
                res = dset[source_slice]
        else:
            res = dset[()]

        return xp.asarray(res) if self.ctx.is_gpu else res

    def write_stacked(
        self, file: h5py.File, name: str, data: xp.ndarray | float
    ) -> None:
        """Simultaneously writes a local array or scalar along each thread/rank.

        The resulting dataset on disk will have the shape `(comm_size, *data.shape)`.

        Parameters
        ----------
        file : h5py.File
            File where to write the data.
        name : str
            Name of the data-field.
        data : xp.ndarray | float
            Local data (array or scalar) to write for the current rank.

        Notes
        -----
        In MPI mode, this triggers a collective I/O operation. All ranks
        must call this method simultaneously.
        """
        data = xp.asarray(data)

        if not data.flags.c_contiguous:
            data = xp.ascontiguousarray(data)

        if self.ctx.is_gpu:
            data = data.get()

        if name not in file:
            shape = (self.ctx.comm_size, *data.shape)
            dset = file.create_dataset(name, shape=shape, dtype=data.dtype)
        else:
            dset = file[name]

        if self._is_collective(file, is_sliced=True):
            with dset.collective:
                dset[self.ctx.rank, ...] = data
        else:
            dset[self.ctx.rank, ...] = data

    def read_stacked(self, file: h5py.File, name: str) -> xp.ndarray:
        """Read the data specifically assigned to the current MPI rank.

        Parameters
        ----------
        file : h5py.File
            File to be read.
        name : str
            Name of the stacked data-field.

        Returns
        -------
        xp.ndarray
            The local data assigned to the current rank.
        """
        return xp.asarray(file[name][self.ctx.rank, ...])

    def write_dict(
        self,
        file: h5py.File,
        data_dict: dict,
        global_shapes: dict | None = None,
        slices: dict | None = None,
    ) -> None:
        """Write the content of an input dictionary in an HDF5 file.

        Parameters
        ----------
        file : h5py.File
            File on which the data will be written.
        data_dict : dict
            Dictionary containing data arrays to write on disk.
        global_shapes : dict, optional
            Dictionary containing the global dimensions of the buffers.
        slices : dict, optional
            Dictionary containing the slicer objects delimiting the position
            of the local buffers in the global buffers.
        """
        global_shapes = global_shapes or {}
        slices = slices or {}

        for key, data in data_dict.items():
            self.write_array(
                file=file,
                name=key,
                data=data,
                global_shape=global_shapes.get(key),
                dest_slice=slices.get(key),
            )

    def read_dict(
        self,
        file: h5py.File,
        keys: list[str] | None = None,
        slices: dict | None = None,
        out_buffers: dict | None = None,
    ) -> dict:
        """Read a dictionary of variables and local arrays from an HDF5 file.

        Parameters
        ----------
        file : h5py.File
            File to be read.
        keys : list of str, optional
            List of variable names to be read. If None, reads all keys in the file.
        slices : dict, optional
            Dictionary containing the slices of each variable, different on each thread.
        out_buffers : dict, optional
            Dictionary of pre-allocated local buffers to read the data into.

        Returns
        -------
        dict
            Dictionary containing the local value of each requested variable.
        """
        keys = keys or list(file.keys())
        slices = slices or {}
        out_buffers = out_buffers or {}

        results = {}
        for key in keys:
            results[key] = self.read_array(
                file=file,
                name=key,
                source_slice=slices.get(key),
                out=out_buffers.get(key),
            )
        return results

    def write_rng(
        self,
        file: h5py.File,
        rng: np.random.Generator | torch.Generator,
    ) -> None:
        """Write the state of a random number generator to an HDF5 file.

        Serializes the given generator's state and writes it into an ``'rng'``
        group within the file, stacked across MPI ranks so that each worker's
        state is preserved independently.

        Parameters
        ----------
        file : h5py.File
            Handle to the open HDF5 file where the state will be written.
        rng : np.random.Generator or torch.Generator
            The random number generator instance to serialize and write.

        Notes
        -----
        In MPI mode, this triggers a collective I/O operation via `write_stacked`.
        All ranks must call this method simultaneously.
        """
        schema = serialize_rng(rng)

        for key, byte_array in schema.items():
            self.write_stacked(file, name=f"rng/{key}", data=byte_array)

    def read_rng(self, file: h5py.File) -> np.random.Generator | torch.Generator:
        """Read and restore the random number generator for the current rank.

        Reads the serialized RNG state schema from the ``'rng'`` group in the file,
        extracting the specific row assigned to the current MPI rank, and reconstructs
        the active generator instance.

        Parameters
        ----------
        file : h5py.File
            Handle to the open HDF5 file containing the written state.

        Returns
        -------
        np.random.Generator or torch.Generator
            The fully restored random number generator, ready for use on the
            current worker.

        Raises
        ------
        KeyError
            If the ``'rng'`` group is missing from the HDF5 file.
        """
        if "rng" not in file:
            raise KeyError("No 'rng' group found in the HDF5 file.")

        schema = {}
        for key in file["rng"]:
            val = self.read_stacked(file, name=f"rng/{key}")
            if self.ctx.is_gpu:
                val = val.get()
            schema[key] = val

        return restore_rng(schema)

    def _is_collective(self, file: h5py.File, is_sliced: bool = True) -> bool:
        """Determines if a collective MPI I/O context should be used.

        Collective operations require an active MPI context, a file opened
        with the MPIO driver, and a sliced/subset operation (entire array
        replacements/reads bypass collective blocks).
        """
        return self.ctx.is_mpi and getattr(file, "driver", None) == "mpio" and is_sliced
