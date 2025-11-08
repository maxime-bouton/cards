"Class that mutualize communication on a shared buffer for linear operator (direct operation only)"

# TODO: cleanse comment

import numpy as np  #! -> change to bm.xp

from mpi4py import MPI
from mcmc.communicator.sync_cartesian_communicator import SyncCartesianCommunicator
from mcmc.backend import xp


class Shared_Communicator:
    def __init__(
        self,
        comm: MPI.Comm,
        grid_size: np.ndarray,
        buffer_size: list[int],
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
        for op in self.operators.keys():
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
