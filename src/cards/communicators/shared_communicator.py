r"""Class triggering communications over a Cartesian pattern shared by a collection of distributed operators.
Entries are received once on a single shared buffer, from which each operator retrieves the required entries for distributed computations.
"""

# TODO: add comments and documentation

import numpy as np
from mpi4py import MPI

import cards.backend as xp
from cards.communicators.sync_cartesian_communicator import SyncCartesianCommunicator


class SharedCommunicator:
    def __init__(
        self,
        comm: MPI.Comm,
        grid_size: np.ndarray,
        buffer_size: np.ndarray,
        operators: dict,
    ):
        self.operators = operators
        self.buffer_size = buffer_size

        self.recv_size = {}
        self.send_size = {}
        self.slice = {}

        for key in self.operators.keys():
            self.recv_size[key] = self.operators[key].get_recv_size()
            self.send_size[key] = self.operators[key].get_send_size()

        self.max_send_size = np.zeros_like(buffer_size)
        self.max_recv_size = np.zeros_like(buffer_size)
        for key in self.operators.keys():
            self.max_send_size = np.maximum(self.max_send_size, self.send_size[key])
            self.max_recv_size = np.maximum(self.max_recv_size, self.recv_size[key])

        self.shared_comm = SyncCartesianCommunicator(
            comm,
            grid_size,
            buffer_size,
            self.max_send_size,
            self.max_recv_size,
            backward=False,
        )

        for key in self.operators.keys():
            self.slice[key] = tuple(
                [
                    np.s_[
                        : self.shared_comm.cartslicer.tile_size[d]
                        + self.recv_size[key][d]
                    ]
                    for d in range(len(buffer_size))
                ]
            )

        self.shared_buffer = xp.zeros(self.shared_comm.cartslicer.facet_size)

    def update_buffer(self, buffer):
        assert (np.asarray(buffer.shape) == self.shared_comm.cartslicer.tile_size).all
        self.shared_buffer[self.shared_comm.cartslicer.slice_facet_to_tile] = buffer
        self.shared_comm.update_borders(self.shared_buffer)

    def apply_operator(self, key, *args):
        return self.operators[key].forward_no_comm(
            self.shared_buffer[self.slice[key]], *args
        )
