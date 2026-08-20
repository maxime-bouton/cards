r"""Class triggering communications over a Cartesian pattern shared by a collection of distributed operators.
Entries are received once on a single shared buffer, from which each operator retrieves the required entries for distributed computations.
"""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

# TODO: add comments and documentation
# TODO: check typing (xp.ndarray or np.ndarray)

import numpy as np
from mpi4py import MPI

import cards.backend as xp
from cards.communicators.sync_cartesian_communicator import SyncCartesianCommunicator


class SharedCommunicator:
    r"""Class gathering and triggering communications following a common communication pattern over the same input array.
    A single communication buffer is thus shared between the different operators, with only appropriate entries recovered to implement each single operator in a distributed setting.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Underlying MPI communicator.
    grid_size : np.ndarray[int]
        Shape of the communication grid along each axis of the problem, as
        returned by ``np.array(MPI.Compute_dims(size, ndims), dtype="i")``.
    buffer_size : np.ndarray[int], of size ``d``
        Size of the ``d`` dimensional buffer decomposed over the Cartesian grid of workers considered. The number of elements handled by the
        current process is computed by an instance of the
        :class:`cards.slicer.cartesian_comm_slicer.CartesianCommSlicer`
        class.
    operators : dict
        Dictionary of operators acting over a common input array.

    Attributes
    ----------
    operators
        Dictionary of operators acting over a common input array.
    buffer_size
        Size of the ``d`` dimensional buffer decomposed over the Cartesian grid of workers considered. The number of elements handled by the
        current process is computed by an instance of the
        :class:`cards.slicer.cartesian_comm_slicer.CartesianCommSlicer`
        class.
    _max_send_size : np.ndarray[int]
        Largest extent of the buffer to be sent to a neighbor worker across each axis.
    _max_recv_size : np.ndarray[int]
        Largest extent of the buffer to be received from a neighbor worker across each axis.
    _slice : dict
        Dictionary of slices corresponding to each operator. Each slice allows
        a portion of the shared buffer to be retrieved for the corresponding
        operator.
    """

    def __init__(
        self,
        comm: MPI.Comm,
        grid_size: np.ndarray,
        buffer_size: np.ndarray,
        operators: dict,
    ) -> None:
        self.operators = operators
        self.buffer_size = buffer_size

        self._max_send_size = np.zeros_like(buffer_size)
        self._max_recv_size = np.zeros_like(buffer_size)
        for key in self.operators.keys():
            self._max_send_size = np.maximum(
                self._max_send_size, self.operators[key].get_send_size()
            )
            self._max_recv_size = np.maximum(
                self._max_recv_size, self.operators[key].get_recv_size()
            )

        self.direct_communicator = SyncCartesianCommunicator(
            comm,
            grid_size,
            buffer_size,
            self._max_send_size,
            self._max_recv_size,
            backward=False,
        )

        self._slice = {}
        for key in self.operators.keys():
            self._slice[key] = tuple(
                [
                    np.s_[
                        : self.direct_communicator.cartslicer.tile_size[d]
                        + self.operators[key].get_recv_size()[d]
                    ]
                    for d in range(len(buffer_size))
                ]
            )
        self.shared_buffer = xp.zeros(self.direct_communicator.cartslicer.facet_size)

    def update_buffer(self, buffer: xp.ndarray) -> None:
        r"""Communicate the borders of an input array tessellated across
        different workers prior to locally evaluate the correspoding portion of the operators' output."""
        assert (
            np.asarray(buffer.shape) == self.direct_communicator.cartslicer.tile_size
        ).all
        self.shared_buffer[self.direct_communicator.cartslicer.slice_facet_to_tile] = (
            buffer
        )
        self.direct_communicator.update_borders(self.shared_buffer)

    def apply_operator(self, key: str, *args) -> None:
        r"""Evaluate the output of a single operator onto the underlying input buffer."""
        return self.operators[key].forward_no_comm(
            self.shared_buffer[self._slice[key]], *args
        )
