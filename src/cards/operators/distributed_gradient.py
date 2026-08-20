"""Distributed implementation of the gradient as a linear operator.
The computations can be done either on CPU or GPU depending on the settings.
"""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

# TODO: documentation

from collections.abc import Sequence

import numpy as np
from mpi4py import MPI

import cards.backend as xp
from cards.communicators.sync_cartesian_communicator import SyncCartesianCommunicator
from cards.operators.linear_operator import LinearOperator


class DistributedGradient2d(LinearOperator):
    r"""Synchronous distributed implementation of a 2D discrete gradient operator.

    Parameters
    ----------
    grid_shape : Sequence[int]
        Number of MPI workers along each of axis of the communicator grid.
    comm: mpi4py.MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD.
    enable_internal_buffer : bool, optional
        Flag to enable interal storage of temporary buffers required for communication, by default True.
    dtype : type, optional
        Type of the entries in communicated arrays, by default xp.float64.

    Attributes
    ----------
    dtype : type, optional
        Type of the entries in communicated arrays, by default xp.float64.
    image_size : np.ndarray[int]
        Numpy array created from ``self.image_shape``.
    grid_size : np.ndarray[int]
        Numpy array created from ``grid_shape``.
    comm : mpi4py.MPI.Comm
        Underlying MPI communicator.
    rank : int
        Rank of the current MPI-process.
    ranknd : numpy.ndarray[int]
        Multi-linear rank of the current MPI-process in the Cartesian grid of
        workers (nD setting).
    direct_communicator : SyncCartesianCommunicator
        Synchronous Cartesian communicator responsible for communications involved in the forward operator.
    adjoint_communicator_v : SyncCartesianCommunicator
        Synchronous Cartesian communicator responsible for communications associated to vertical differences involved in the adjoint operator.
    adjoint_communicator_h : SyncCartesianCommunicator
        Synchronous Cartesian communicator responsible for communications associated to horizontal differences involved in the adjoint operator.
    is_first : np.ndarray[bool]
        Flag indicating if current worker is in first position alogn each axis
        of the Cartesian communication grid.
    is_last : np.ndarray[bool]
        Flag indicating if current worker is in first position alogn each axis
        of the Cartesian communication grid.
    forward_buffer : xp.ndarray
        Temporary buffer to receive communications involved in the forward operator.
    adjoint_buffer_h : xp.ndarray
        Temporary buffer to receive communications involved in the adjoint operator (horizontal communicator).
    adjoint_buffer_v : xp.ndarray
        Temporary buffer to receive communications involved in the adjoint operator (vertical communicator).
    """

    def __init__(
        self,
        image_shape: Sequence[int],
        grid_shape: Sequence[int],
        comm: MPI.Comm = MPI.COMM_WORLD,
        enable_internal_buffer: bool = True,
        dtype: type = xp.float64,
    ):
        super().__init__(image_shape, [2, *image_shape])
        self.image_size = np.asarray(self.image_shape)

        self.dtype = dtype
        self.comm = comm
        self.grid_size = np.asarray(grid_shape)

        dim_extension = [0] * (len(self.grid_size) - 2)
        overlap = np.asarray(dim_extension + [1, 1])
        self.direct_communicator = SyncCartesianCommunicator(
            self.comm,
            self.grid_size,
            self.image_size,
            overlap,
            overlap,
            backward=False,
            dtype=self.dtype,
        )
        self.adjoint_communicator_v = SyncCartesianCommunicator(
            self.comm,
            self.grid_size,
            self.image_size,
            np.asarray(dim_extension + [1, 0]),
            np.asarray(dim_extension + [1, 0]),
            backward=True,
            dtype=self.dtype,
        )
        self.adjoint_communicator_h = SyncCartesianCommunicator(
            self.comm,
            self.grid_size,
            self.image_size,
            np.asarray(dim_extension + [0, 1]),
            np.asarray(dim_extension + [0, 1]),
            backward=True,
            dtype=self.dtype,
        )

        # TODO: see if rank/ranknd is also needed on top of self.direct_communicator.rank/ranknd
        self.rank = self.direct_communicator.rank
        self.ranknd = self.direct_communicator.ranknd
        self.grid_size = self.direct_communicator.grid_size

        # TODO: mutualize is_first/last -> communications ?
        self.is_first = self.ranknd == 0
        self.is_last = self.ranknd == self.grid_size - 1

        if enable_internal_buffer:
            self.forward_buffer = xp.zeros(
                self.direct_communicator.cartslicer.facet_size, dtype=self.dtype
            )
        self.adj_buffer = xp.zeros(
            self.direct_communicator.cartslicer.tile_size, dtype=self.dtype
        )
        self.adjoint_buffer_v = xp.zeros(
            self.adjoint_communicator_v.cartslicer.facet_size,
            dtype=self.dtype,
        )
        self.adjoint_buffer_h = xp.zeros(
            self.adjoint_communicator_h.cartslicer.facet_size,
            dtype=self.dtype,
        )

    def _chunk_gradient_2d(self, x: xp.ndarray):
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

        # worker in last position along axis 1 of the grid: no border coming
        # from the next worker

        *c, h, w = x.shape
        local_h = h if self.is_last[-2] else h - 1
        local_w = w if self.is_last[-1] else w - 1
        u = xp.zeros((2, *c, local_h, local_w), dtype=self.dtype)  # x.dtype

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

    def _chunk_gradient_2d_adjoint(self, uh: xp.ndarray, uv: xp.ndarray):
        r"""Chunk of the adjoint 2d discrete gradient.

        Compute a chunk of the adjoint 2d discrete gradient. Assumes backward border overlap between the arrays handled by consecutive worker.

        Parameters
        ----------
        uh : xp.ndarray
            Local chunk of the horizontal difference.
        uv : xp.ndarray
            Local chunk of the vertical difference.

        ..note::
            Backward overlap is expected.
        """
        # TODO: need to check size of u?
        assert len(uh.shape) >= 2 and len(uh.shape) == len(uv.shape), (
            "gradient_2d_adjoint: Invalid input, expected len(uh.shape) == len(uv.shape)"
        )

        # self.adj_buffer = xp.zeros(self.direct_communicator.cartslicer.tile_size, dtype=self.dtype)
        self.adj_buffer.fill(0)

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
                self.adj_buffer += uh[..., :-1] - uh[..., 1:]

    def forward(self, image: xp.ndarray) -> xp.ndarray:
        # NOTE: in this function, ``image`` refers to the local image tile handled by the current process
        *_, h, w = self.direct_communicator.cartslicer.tile_size
        self.forward_buffer[..., :h, :w] = image

        self.direct_communicator.update_borders(self.forward_buffer)

        return self._chunk_gradient_2d(self.forward_buffer)

    def adjoint(self, data: xp.ndarray) -> xp.ndarray:
        # NOTE: in this function, ``data`` refers to the local data tile handled by the current process
        *_, h, w = self.adjoint_communicator_v.cartslicer.tile_size
        self.adjoint_buffer_v[..., -h:, -w:] = data[1]
        self.adjoint_buffer_h[..., -h:, -w:] = data[0]

        self.adjoint_communicator_v.update_borders(self.adjoint_buffer_v)
        self.adjoint_communicator_h.update_borders(self.adjoint_buffer_h)

        self._chunk_gradient_2d_adjoint(
            self.adjoint_buffer_h,
            self.adjoint_buffer_v,
        )
        return self.adj_buffer

    def forward_no_comm(self, image: xp.ndarray):
        # NOTE: in this function, ``image`` refers to the local image tile handled by the current process
        return self._chunk_gradient_2d(image)

    # TODO: add this collection of features through inhteritance, instead of repeating it each time?
    def get_recv_size(self) -> np.ndarray:
        return self.direct_communicator.cartslicer.recv_size

    def get_send_size(self) -> np.ndarray:
        return self.direct_communicator.cartslicer.send_size
