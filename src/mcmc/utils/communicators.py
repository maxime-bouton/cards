"""Set of helper communicator objects to define a common MPI communication
interface across the different samplers considered.
"""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# reference: P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.

import weakref
from abc import ABC, abstractmethod

import numpy as np

import dsgs.utils.communications as libcomm

# Documentation Abstract Base Class (ABC): necessary since no virtual classes
# by default in Python
# https://docs.python.org/3/library/abc.html
# https://stackoverflow.com/questions/4613000/difference-between-cls-and-self-in-python-classes
# https://www.geeksforgeeks.org/class-method-vs-static-method-python/
# Always use self for the first argument to instance methods.
# Always use cls for the first argument to class methods.
# class methods used as utility function, no access to the state of the object

# a nice post about finalizers
# https://docs.python.org/3.6/library/weakref.html#finalizer-objects


# TODO: prepare AsyncCartesianCommunicator (need to communicate separately
# along the diagonals)


class BaseCommunicator(ABC):
    """Base communicator, including the minimal data and methods to define the
    communication process.

    Attributes
    ----------
    comm : mpi4py.MPI.Comm
        Underlying MPI communicator.
    grid_size : numpy.ndarray[int], of size ``d``
        Number of workers along each of the ``d`` dimensions of the
        communicator grid.
    itemsize : numpy.dtype.itemsize
        Size (in bytes) of the scalar type to be handled during the
        communications.
    facet_size : numpy.ndarray[int], of size ``d``
        Number of elements along each of the ``d`` dimensions of the facet
        handled by the current process.
    """

    def __init__(
        self,
        comm,
        grid_size,
        itemsize,
        facet_size,
    ):
        """BaseCommunicator constructor.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            Underlying MPI communicator.
        grid_size : list of int, of size ``d``
            Number of workers along each of the ``d`` dimensions of the
            communicator grid.
        itemsize : numpy.dtype.itemsize
            Size (in bytes) of the scalar type to be handled during the
            communications.
        facet_size : numpy.ndarray[int], of size ``d``
            Number of elements along each of the ``d`` dimensions of the facet
            handled by the current process.

        Raises
        ------
        ValueError
            ``grid_size`` and ``facet_size`` must contain the same number of
            element.
        """

        # primary attributes
        self.comm = comm
        self.grid_size__ = grid_size
        self.grid_size = np.array(self.grid_size__, dtype="i")
        self.itemsize = itemsize
        if not self.grid_size.size == facet_size.size:
            raise ValueError(
                "`grid_size` and `facet_size` must contain the same number of element."
            )
        self.facet_size = facet_size

        # secondary attributes
        self.ndims = len(grid_size)
        self.rank = self.comm.Get_rank()

    @abstractmethod
    def update_borders(self, local_array):  # pragma: no cover
        """Update the borders of a local array using the communication scheme
        defined in the communicator.

        Parameters
        ----------
        local_array : numpy.ndarray, with ``d`` dimensions
            Local array to be updated through communications.

        Returns
        -------
        NotImplemented

        Note
        ----
        The method needs to be implemented in any class inheriting from
        BaseCommunicator.
        """
        return NotImplemented

    @abstractmethod
    def remove(self):  # pragma: no cover
        """Base function to clean up auxiliary quantities when the object can be
        safely deleted.

        Returns
        -------
        NotImplemented
        """
        return NotImplemented


def free_custom_mpi_types(resizedsendsubarray, resizedrecvsubarray, isvalid_comm):
    r"""Freeing custom MPI resized types.

    Parameters
    ----------
    resizedsendsubarray : list of mpi4py.MPI.Datatype, of size ``d``
        Custom MPI subarray type describing the data sent by the current
        process, as returned by ``mpi4py.MPI.Datatype.Create_subarray``.
    resizedrecvsubarray : list of mpi4py.MPI.Datatype, of size ``d``
        Custom MPI subarray type describing the data received by the current
        process, as returned by ``mpi4py.MPI.Datatype.Create_subarray``.
    isvalid_comm : numpy.ndarray of bool, of size ``d``
        Boolean vector indicating wether each of the ``d`` possible
        communications are valid for the current worker (e.g., absence of
        neighbour, ...).
    """

    ndims = isvalid_comm.size

    # free custom MPI types
    for d in range(ndims):
        if isvalid_comm[d]:
            resizedsendsubarray[d].Free()
            resizedrecvsubarray[d].Free()


class SyncCartesianCommunicator(BaseCommunicator):
    """Cartesian communicator underlying the synchronous samplers.

    Attributes
    ----------
    comm : mpi4py.MPI.Comm
        Underlying MPI communicator.
    grid_size : list of int, of size ``d``
        Number of workers along each of the ``d`` dimensions of the
        communicator grid.
    itemsize : numpy.dtype.itemsize
        Size (in bytes) of the scalar type to be handled during the
        communications.
    facet_size : numpy.ndarray[int], of size ``d``
        Number of elements along each of the ``d`` dimensions of the facet
        handled by the current process.
    overlap_size : numpy.ndarray[int], of size ``d``
        Size of the overlap between the array handled by two different
        workers.
    circular_boundaries : bool
        Indicates whether periodic boundary conditions need to be
        considered for the communicator grid along each axis. By default
    backward : bool, optional
        Direction of the overlap between facets along all the axis (True
        for backward overlap, False for forward overlap). By default False.
    """

    def __init__(
        self,
        comm,
        cartcomm,
        grid_size,
        itemsize,
        facet_size,
        overlap_size,
        backward=False,
    ):
        """SyncCartesianCommunicator constructor.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            Underlying MPI communicator.
        cartcomm : mpi4py.MPI.Cartcomm
            Underlying Cartesian MPI communicator.
        grid_size : list of int, of size ``d``
            Number of workers along each of the ``d`` dimensions of the
            communicator grid.
        itemsize : numpy.dtype.itemsize
            Size (in bytes) of the scalar type to be handled during the
            communications.
        facet_size : numpy.ndarray[int], of size ``d``
            Number of elements along each of the ``d`` dimensions of the facet
            handled by the current process.
        overlap_size : numpy.ndarray[int], of size ``d``
            Size of the overlap between the array handled by two different
            workers.
        backward : bool, optional
            Direction of the overlap between facets along all the axis (True
            for backward overlap, False for forward overlap). By default False.

        Raises
        ------
        ValueError
            ``overlap_size`` and ``grid_size`` must contain the same number of
            element.
        """
        super(SyncCartesianCommunicator, self).__init__(
            comm,
            grid_size,
            itemsize,
            facet_size,
        )
        self.overlap_size = overlap_size

        # TODO: give the possibility to set the type from the interface
        # configure communication scheme
        (
            self.dest,
            self.src,
            self.resizedsendsubarray,
            self.resizedrecvsubarray,
        ) = libcomm.setup_border_update(
            cartcomm,
            self.ndims,
            self.itemsize,
            self.facet_size,
            self.overlap_size,
            backward=backward,
        )

        # (
        #     self.dest,
        #     self.src,
        #     self.resizedsendsubarray,
        #     self.resizedrecvsubarray,
        # ) = libcomm.setup_border_update_int(
        #     cartcomm,
        #     self.ndims,
        #     self.itemsize,
        #     self.facet_size,
        #     self.overlap_size,
        #     backward=backward,
        # )

        # setup finalizer
        self.isvalid_comm = self.overlap_size > 0
        self._finalizer = weakref.finalize(
            self,
            free_custom_mpi_types,
            self.resizedsendsubarray,
            self.resizedrecvsubarray,
            self.isvalid_comm,
        )

    def update_borders(self, local_array):
        """Update the borders of a local array using the communication scheme
        defined in the communicator.

        Parameters
        ----------
        local_array : numpy.ndarray, with ``d`` dimensions, float entries
            Local array to be updated through communications.

        Note
        ----
        - The input array, ``local_array`` is updated in-place.
        - The function will trigger a segfault error if the type of
          ``local_array`` is different from ``float`` (due to the hard-coded
          MPI type used to defined ``self.resizedsendsubarray`` and
          ``self.resizedrecvsubarray``)
        """
        for d in range(self.ndims):
            self.comm.Sendrecv(
                [local_array, 1, self.resizedsendsubarray[d]],
                self.dest[d],
                recvbuf=[local_array, 1, self.resizedrecvsubarray[d]],
                source=self.src[d],
            )
        return

    def remove(self):
        """Trigger object finalizer (clean-up)."""
        return self._finalizer()

    @property
    def removed(self):
        """Check whether the object has been finalized."""
        return not self._finalizer.alive


class SyncCartesianCommunicatorTV(BaseCommunicator):
    """Cartesian communicator underlying synchronous discrete TV computations.

    Attributes
    ----------
    comm : mpi4py.MPI.Comm
        Underlying MPI communicator.
    grid_size : list of int, of size ``d``
        Number of workers along each of the ``d`` dimensions of the
        communicator grid.
    itemsize : numpy.dtype.itemsize
        Size (in bytes) of the scalar type to be handled during the
        communications.
    facet_size : numpy.ndarray[int], of size ``d``
        Number of elements along each of the ``d`` dimensions of the facet
        handled by the current process.
    overlap_size : numpy.ndarray[int], of size ``d``
        Size of the overlap between the array handled by two different
        workers.
    circular_boundaries : bool
        Indicates whether periodic boundary conditions need to be
        considered for the communicator grid along each axis. By default
    backward : bool, optional
        Direction of the overlap between facets along all the axis (True
        for backward overlap, False for forward overlap). By default False.

    Note
    ----
    Try to merge this object with
    :class:`dsgs.utils.SyncCartesianCommunicator`?
    """

    def __init__(
        self,
        comm,
        cartcomm,
        grid_size,
        itemsize,
        facet_size,
        backward=False,
    ):
        """SyncCartesianCommunicatorTV constructor.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            Underlying MPI communicator.
        cartcomm : mpi4py.MPI.Cartcomm
            Underlying Cartesian MPI communicator.
        grid_size : list of int, of size ``d``
            Number of workers along each of the ``d`` dimensions of the
            communicator grid.
        itemsize : numpy.dtype.itemsize
            Size (in bytes) of the scalar type to be handled during the
            communications.
        facet_size : numpy.ndarray[int], of size ``d``
            Number of elements along each of the ``d`` dimensions of the facet
            handled by the current process.
        overlap_size : numpy.ndarray[int], of size ``d``
            Size of the overlap between the array handled by two different
            workers.
        backward : bool, optional
            Direction of the overlap between facets along all the axis (True
            for backward overlap, False for forward overlap). By default False.

        Raises
        ------
        ValueError
            ``overlap_size`` and ``grid_size`` must contain the same number of
            element.
        """
        super(SyncCartesianCommunicatorTV, self).__init__(
            comm,
            grid_size,
            itemsize,
            facet_size,
        )
        self.overlap_size = np.ones(facet_size.size, dtype="i")

        # configure communication scheme
        (
            self.dest,
            self.src,
            self.resizedsendsubarray,
            self.resizedrecvsubarray,
        ) = libcomm.setup_border_update_tv(
            cartcomm,
            self.ndims,
            self.itemsize,
            self.facet_size,
            self.overlap_size,
            backward=backward,
        )

        # setup finalizer
        self.isvalid_comm = self.overlap_size > 0
        # np.logical_and(self.overlap_size > 0, self.grid_size > 1)
        self._finalizer = weakref.finalize(
            self,
            free_custom_mpi_types,
            self.resizedsendsubarray,
            self.resizedrecvsubarray,
            self.isvalid_comm,
        )

    def update_borders(self, local_array):
        """Update the borders of a local array using the communication scheme
        defined in the communicator.

        Parameters
        ----------
        local_array : numpy.ndarray, with ``d`` dimensions, float entries
            Local array to be updated through communications.

        Note
        ----
        - The input array, ``local_array`` is updated in-place.
        - The function will trigger a segfault error if the type of
          ``local_array`` is different from ``float`` (due to the hard-coded
          MPI type used to defined ``self.resizedsendsubarray`` and
          ``self.resizedrecvsubarray``)
        """
        for d in range(self.ndims):
            self.comm.Sendrecv(
                [local_array, 1, self.resizedsendsubarray[d]],
                self.dest[d],
                recvbuf=[local_array, 1, self.resizedrecvsubarray[d]],
                source=self.src[d],
            )
        return

    def remove(self):
        """Trigger object finalizer (clean-up)."""
        return self._finalizer()

    @property
    def removed(self):
        """Check whether the object has been finalized."""
        return not self._finalizer.alive
