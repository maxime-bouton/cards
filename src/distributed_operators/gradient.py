import numpy as np
from mpi4py import MPI

from slicer.cartesian_comm_slicer import CartesianCommSlicer
from communicator.sync_cartesian_communicator import SyncCartesianCommunicator

def chunk_gradient_2d(x, islast):
    r"""Chunk of the 2d discrete gradient (with jit support).

    Compute a chunk of the 2d discrete gradient operator (using jit
    compilation). Assumes forward border overlap between the arrays handled by
    consecutive worker.

    Parameters
    ----------
    x : numpy.ndarray[float64 or complex128], 2d
        Input array including border for forwrd overlap.
    islast : numpy.ndarray, bool, 1d
        Vector indicating whether the chunk is the last one along each
        dimension of the Cartesian process grid.

    Returns
    -------
    u : numpy.ndarray[float64 or complex128], 2d
        Local chunk of the horizontal and vertical differences.
    """
    assert (
        len(x.shape) == 2 and islast.size == 2
    ), "gradient_2d: Invalid input, expected len(x.shape)==len(islast.shape)==2"

    # worker in last position along axis 1 of the grid: no border coming from
    # the next worker
    if islast[0]:
        local_shape0 = x.shape[0]
    else:
        local_shape0 = x.shape[0] - 1
    if islast[1]:
        local_shape1 = x.shape[1]
    else:
        local_shape1 = x.shape[1] - 1
    u = np.zeros((2, local_shape0, local_shape1), dtype=x.dtype)

    # horizontal differences uh = u[0, :, :]
    if islast[1]:
        if islast[0]:
            # uh = np.zeros(x.shape, dtype=x.dtype)
            u[0, :, :-1] = x[:, 1:] - x[:, :-1]
        else:
            # uh = np.zeros((x.shape[0] - 1, x.shape[1]), dtype=x.dtype)
            u[0, :, :-1] = x[:-1, 1:] - x[:-1, :-1]
    else:
        if islast[0]:
            u[0] = x[:, 1:] - x[:, :-1]
        else:
            # ! when there is a border for both axes, need to discard from x
            # the border along the axis not considered for the difference
            u[0] = x[:-1, 1:] - x[:-1, :-1]

    # vertical differences: uv = u[1, :, :]
    if islast[0]:
        if islast[1]:
            # uv = np.zeros(x.shape, dtype=x.dtype)
            u[1, :-1, :] = x[1:, :] - x[:-1, :]
        else:
            # uv = np.zeros((x.shape[0], x.shape[1] - 1), dtype=x.dtype)
            u[1, :-1, :] = x[1:, :-1] - x[:-1, :-1]
    else:
        if islast[1]:
            u[1] = x[1:, :] - x[:-1, :]
        else:
            # ! when there is a border for both axes, need to discard from x
            # the border along the axis not considered for the difference
            u[1] = x[1:, :-1] - x[:-1, :-1]

    return u


def chunk_gradient_2d_adjoint(uh, uv, x, isfirst, islast):
    r"""Chunk of the adjoint 2d discrete gradient (with jit support).

    Compute a chunk of the adjoint 2d discrete gradient. Assumes backward border overlap between the arrays handled by consecutive worker.
    Expects the buffer given in entry to be set to 0.

    Parameters
    ----------
    uh : numpy.ndarray[float64 or complex128], 2d
        Local chunk of the horizontal difference.
    uv : numpy.ndarray[float64 or complex128], 2d
        Local chunk of the vertical difference.
    x : numpy.ndarray[float64 or complex128], 2d
        Output array (updated in-place).
    isfirst : numpy.ndarray, bool, 1d
        Vector indicating whether the chunk is the first one along each
        dimension of the Cartesian process grid.
    islast : numpy.ndarray, bool, 1d
        Vector indicating whether the chunk is the last one along each
        dimension of the Cartesian process grid.

    ..note::
        The array ``x`` is updated in-place. Backward overlap is expected.
    """
    # TODO: need to check size of u?
    assert (
        len(uh.shape) == len(uv.shape) == 2
    ), "gradient_2d_adjoint: Invalid input, expected len(uh.shape) == len(uv.shape) == 2"

    # vertical: uv = u[1, :, :]
    if isfirst[0]:  # no overlap along axis 0
        x[0, :] -= uv[0, :]
        if islast[0]:
            x[1:-1, :] += uv[:-2, :] - uv[1:-1, :]
            x[-1, :] += uv[-2, :]
        else:
            x[1:, :] += uv[:-1, :] - uv[1:, :]
    else:
        if islast[0]:
            x[:-1, :] += uv[:-2, :] - uv[1:-1, :]
            x[-1, :] += uv[-2, :]
        else:
            x += uv[:-1, :] - uv[1:, :]

    # horizontal: uh = u[0, :, :]
    if isfirst[1]:  # no overlap along axis 0
        x[:, 0] -= uh[:, 0]
        if islast[1]:
            x[:, 1:-1] += uh[:, :-2] - uh[:, 1:-1]
            x[:, -1] += uh[:, -2]
        else:
            x[:, 1:] += uh[:, :-1] - uh[:, 1:]
    else:
        if islast[1]:
            x[:, :-1] += uh[:, :-2] - uh[:, 1:-1]
            x[:, -1] += uh[:, -2]
        else:
            x += uh[:, :-1] - uh[:, 1:]

    return

class distributed_gradient2d(  ):
    def __init__(self, global_size : np.ndarray, grid_size : np.ndarray ) -> None:
        
        overlap = np.asarray([1,1])
        self.cart_comm = SyncCartesianCommunicator( MPI.COMM_WORLD, grid_size, global_size, overlap, overlap, backward=False )
        self.adj_cart_comm_v = SyncCartesianCommunicator( MPI.COMM_WORLD, grid_size, global_size, np.asarray([1,0]),np.asarray([1,0]), backward=True )
        self.adj_cart_comm_h = SyncCartesianCommunicator( MPI.COMM_WORLD, grid_size, global_size, np.asarray([0,1]),np.asarray([0,1]), backward=True )

        self.local_buffer = np.zeros(self.cart_comm.cartslicer.facet_size)
        self.local_buffer_adj_v = np.zeros(self.adj_cart_comm_v.cartslicer.facet_size)
        self.local_buffer_adj_h = np.zeros(self.adj_cart_comm_h.cartslicer.facet_size)
        
    def compute_grad(self, local_data : np.ndarray) -> np.ndarray :
        [m,n] = self.cart_comm.cartslicer.tile_size
        self.local_buffer[:m,:n] = local_data.copy()

        self.cart_comm.update_borders(self.local_buffer)

        grid_size = self.cart_comm.cartslicer.grid_size
        ranknd = self.cart_comm.ranknd
        is_border = np.asarray( [ranknd[0] == (grid_size[0]-1), ranknd[1] == (grid_size[1]-1)  ] )
        return chunk_gradient_2d(self.local_buffer, is_border)

    def compute_adjoint(self,res : np.ndarray , local_data_h : np.ndarray, local_data_v : np.ndarray) -> None :
        [m,n] = self.adj_cart_comm_v.cartslicer.tile_size
        self.local_buffer_adj_v[-m:,-n:] = local_data_v.copy()
        self.local_buffer_adj_h[-m:,-n:] = local_data_h.copy()

        self.adj_cart_comm_v.update_borders(self.local_buffer_adj_v)
        self.adj_cart_comm_h.update_borders(self.local_buffer_adj_h)

        grid_size = self.cart_comm.grid_size
        ranknd = self.cart_comm.ranknd
        is_first = np.asarray([ ranknd[0] == 0 , ranknd[1] == 0  ])
        is_last = np.asarray( [ ranknd[0] == (grid_size[0]-1), ranknd[1] == (grid_size[1]-1) ] )

        chunk_gradient_2d_adjoint(self.local_buffer_adj_h, self.local_buffer_adj_v, res, is_first, is_last)
        return 

