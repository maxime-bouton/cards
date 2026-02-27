"""Distributed implementation of the gradient as a linear operator.
The computations can be done either on CPU or GPU depending on the settings.
"""

import numpy as np
from mpi4py import MPI

from mcmc.backend import xp
from mcmc.communicator.sync_cartesian_communicator import SyncCartesianCommunicator


class MpiGradient2d:
    def __init__(
        self,
        global_size: np.ndarray,
        grid_size: np.ndarray,
        comm: MPI.Comm,
        dtype: xp.dtype = xp.float64,
    ) -> None:
        self.comm = comm

        dim_extension = [0] * (len(grid_size) - 2)
        overlap = np.asarray(dim_extension + [1, 1])
        self.cart_comm = SyncCartesianCommunicator(
            self.comm, grid_size, global_size, overlap, overlap, backward=False
        )
        self.adj_cart_comm_v = SyncCartesianCommunicator(
            self.comm,
            grid_size,
            global_size,
            np.asarray(dim_extension + [1, 0]),
            np.asarray(dim_extension + [1, 0]),
            backward=True,
            dtype=dtype,
        )
        self.adj_cart_comm_h = SyncCartesianCommunicator(
            self.comm,
            grid_size,
            global_size,
            np.asarray(dim_extension + [0, 1]),
            np.asarray(dim_extension + [0, 1]),
            backward=True,
            dtype=dtype,
        )

        self.rank = self.cart_comm.rank
        grid_size = self.cart_comm.grid_size
        self.ranknd = self.cart_comm.ranknd
        # TODO mutualize is_first/last -> communications ?
        self.is_first = self.ranknd == 0
        self.is_last = self.ranknd == grid_size - 1
        self.adj_shape = self.cart_comm.cartslicer.tile_size

        self.local_buffer = xp.zeros(self.cart_comm.cartslicer.facet_size, dtype=dtype)
        self.adj_buffer = xp.zeros(self.cart_comm.cartslicer.tile_size, dtype=dtype)
        self.local_buffer_adj_v = xp.zeros(
            self.adj_cart_comm_v.cartslicer.facet_size,
            dtype=dtype,
        )
        self.local_buffer_adj_h = xp.zeros(
            self.adj_cart_comm_h.cartslicer.facet_size,
            dtype=dtype,
        )

    def chunk_gradient_2d(self, x: xp.ndarray):
        r"""Chunk of the 2d discrete gradient .

        Compute a chunk of the 2d discrete gradient operator. Assumes forward border overlap between the arrays handled by
        consecutive worker.

        Parameters
        ----------
        x : xp.ndarray[float64 or complex128], 2d
            Input array including border for forwrd overlap.
        Returns
        -------
        u : xp.ndarray[float64 or complex128], 2d
            Local chunk of the horizontal and vertical differences.
        """

        assert len(x.shape) >= 2 and self.is_last.size == len(x.shape), (
            "gradient_2d: Invalid input, expected len(x.shape)==len(is_last.shape)"
        )

        # worker in last position along axis 1 of the grid: no border coming from
        # the next worker

        *c, h, w = x.shape
        local_h = h if self.is_last[-2] else h - 1
        local_w = w if self.is_last[-1] else w - 1
        u = xp.zeros((2, *c, local_h, local_w), dtype=x.dtype)

        # horizontal differences uh = u[0, :, :]
        if self.is_last[-1]:
            if self.is_last[-2]:
                # uh = np.zeros(x.shape, dtype=x.dtype)
                u[0, ..., :-1] = x[..., 1:] - x[..., :-1]
            else:
                # uh = np.zeros((x.shape[0] - 1, x.shape[1]), dtype=x.dtype)
                u[0, ..., :-1] = x[..., :-1, 1:] - x[..., :-1, :-1]
        else:
            if self.is_last[-2]:
                u[0] = x[..., 1:] - x[..., :-1]
            else:
                # ! when there is a border for both axes, need to discard from x
                # the border along the axis not considered for the difference
                u[0] = x[..., :-1, 1:] - x[..., :-1, :-1]

        # vertical differences: uv = u[1, :, :]
        if self.is_last[-2]:
            if self.is_last[-1]:
                # uv = np.zeros(x.shape, dtype=x.dtype)
                u[1, ..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
            else:
                # uv = np.zeros((x.shape[0], x.shape[1] - 1), dtype=x.dtype)
                u[1, ..., :-1, :] = x[..., 1:, :-1] - x[..., :-1, :-1]
        else:
            if self.is_last[-1]:
                u[1] = x[..., 1:, :] - x[..., :-1, :]
            else:
                # ! when there is a border for both axes, need to discard from x
                # the border along the axis not considered for the difference
                u[1] = x[..., 1:, :-1] - x[..., :-1, :-1]

        return u

    def chunk_gradient_2d_adjoint(self, uh: xp.ndarray, uv: xp.ndarray) -> xp.array:
        r"""Chunk of the adjoint 2d discrete gradient.

        Compute a chunk of the adjoint 2d discrete gradient. Assumes backward border overlap between the arrays handled by consecutive worker.

        Parameters
        ----------
        uh : cupy.ndarray[float64 or complex128], 2d
            Local chunk of the horizontal difference.
        uv : cupy.ndarray[float64 or complex128], 2d
            Local chunk of the vertical difference.

        ..note::
            The array ``x`` is updated in-place. Backward overlap is expected.
        """
        # TODO: need to check size of u?
        assert len(uh.shape) >= 2 and len(uh.shape) == len(uv.shape), (
            "gradient_2d_adjoint: Invalid input, expected len(uh.shape) == len(uv.shape)"
        )

        # self.adj_buffer.fill(0) #! cupy memory management does not allow that
        self.adj_buffer = xp.zeros(self.adj_shape)

        # vertical: uv = u[1, :, :, :]
        if self.is_first[-2]:  # no overlap along axis 0
            self.adj_buffer[..., 0, :] -= uv[..., 0, :]
            if self.is_last[-2]:
                self.adj_buffer[..., 1:-1, :] += uv[..., :-2, :] - uv[..., 1:-1, :]
                self.adj_buffer[..., -1, :] += uv[..., -2, :]
            else:
                self.adj_buffer[..., 1:, :] += uv[..., :-1, :] - uv[..., 1:, :]
        else:
            if self.is_last[-2]:
                self.adj_buffer[..., :-1, :] += uv[..., :-2, :] - uv[..., 1:-1, :]
                self.adj_buffer[..., -1, :] += uv[..., -2, :]
            else:
                self.adj_buffer[..., :, :] += uv[..., :-1, :] - uv[..., 1:, :]

        # horizontal: uh = u[0, :, :, :]
        if self.is_first[-1]:  # no overlap along axis 0
            self.adj_buffer[..., 0] -= uh[..., 0]
            if self.is_last[-1]:
                self.adj_buffer[..., 1:-1] += uh[..., :-2] - uh[..., 1:-1]
                self.adj_buffer[..., -1] += uh[..., -2]
            else:
                self.adj_buffer[..., 1:] += uh[..., :-1] - uh[..., 1:]
        else:
            if self.is_last[-1]:
                self.adj_buffer[..., :-1] += uh[..., :-2] - uh[..., 1:-1]
                self.adj_buffer[..., -1] += uh[..., -2]
            else:
                self.adj_buffer[..., :] += uh[..., :-1] - uh[..., 1:]

        return

    def forward(self, local_data: xp.ndarray) -> xp.ndarray:
        *_, h, w = self.cart_comm.cartslicer.tile_size
        self.local_buffer[..., :h, :w] = local_data

        self.cart_comm.update_borders(self.local_buffer)

        return self.chunk_gradient_2d(self.local_buffer)

    def adjoint(self, local_data: xp.ndarray) -> xp.ndarray:
        *_, h, w = self.adj_cart_comm_v.cartslicer.tile_size
        self.local_buffer_adj_v[..., -h:, -w:] = local_data[1]
        self.local_buffer_adj_h[..., -h:, -w:] = local_data[0]

        self.adj_cart_comm_v.update_borders(self.local_buffer_adj_v)
        self.adj_cart_comm_h.update_borders(self.local_buffer_adj_h)

        self.chunk_gradient_2d_adjoint(
            self.local_buffer_adj_h,
            self.local_buffer_adj_v,
        )
        return self.adj_buffer
