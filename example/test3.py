import numpy as np

from mpi4py import MPI

from slicer.comm_slicer import CommSlicer
from slicer.cartesian_comm_slicer import CartesianCommSlicer
from communicator.sync_cartesian_communicator import SyncCartesianCommunicator
from operators.jtv import  gradient_2d_adjoint


def chunk_gradient_2d_adjoint(uh, uv, x, isfirst, islast):
    r"""Chunk of the adjoint 2d discrete gradient (with jit support).

    Compute a chunk of the adjoint 2d discrete gradient. Assumes backward border overlap between the arrays handled by consecutive worker.

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


if __name__ == '__main__' :

    M = 12
    N = 12
    global_size = np.asarray([M,N])

    comm = MPI.COMM_WORLD
   
    rank  = comm.Get_rank()
    grid_dims = np.asarray( MPI.Compute_dims(comm.Get_size(), 2) ,dtype = int )
    m = M//grid_dims[0]
    n = N//grid_dims[1]


    cart_comm = comm.Create_cart(dims = grid_dims)

    cartesian_comm_rows = SyncCartesianCommunicator(comm, grid_dims, global_size, np.asarray([1,0]), np.asarray([1,0]), backward=True)
    cartesian_comm_cols = SyncCartesianCommunicator(comm, grid_dims, global_size, np.asarray([0,1]), np.asarray([0,1]), backward=True)

    X = np.zeros([2,M,N])
    local = np.zeros([2,m+1,n+1])
    local_v = np.zeros([m+1,n+1])
    local_h = np.zeros([m+1,n+1])

    if rank == 0 :
        X= np.random.rand(2,M,N)
    
    cart_comm.Bcast([X, MPI.DOUBLE], root=0)

    grid_coords = cartesian_comm_cols.cartslicer.ranknd

    first_x = ( 0 == grid_coords[0])
    first_y = ( 0 == grid_coords[1])

    last_x = (grid_dims[0]-1 == grid_coords[0])
    last_y = (grid_dims[1]-1 == grid_coords[1])

    local_v = np.zeros(cartesian_comm_rows.cartslicer.facet_size)
    local_h = np.zeros(cartesian_comm_cols.cartslicer.facet_size)

    x = cartesian_comm_rows.cartslicer.tile_range[0][0]
    y = cartesian_comm_rows.cartslicer.tile_range[1][0]

    local_h[-m:,-n:] = X[0][x:x+m,y:y+n]
    local_v[-m:,-n:] = X[1][x:x+m,y:y+n]
   
    cartesian_comm_rows.update_borders(local_v)
    cartesian_comm_cols.update_borders(local_h)

    chunk_adj = np.zeros(cartesian_comm_rows.cartslicer.tile_size)
    
    chunk_gradient_2d_adjoint(local_h, local_v, chunk_adj, np.asarray([first_x,first_y]), np.asarray([last_x,last_y]) )


    if rank == 0 :
        global_adj = gradient_2d_adjoint(X[0],X[1])

    distributed_adj = np.zeros([M,N])

    comm.Send([chunk_adj, MPI.DOUBLE], 0)

    if rank == 0 :
        for i in range( comm.Get_size() ):
            comm.Recv([chunk_adj,MPI.DOUBLE], i )
            coord = cart_comm.Get_coords(i)
            x = coord[0]*m
            y = coord[1]*n
            distributed_adj[ x:x+m , y:y+n ] = chunk_adj[-m:,-n:].copy()

        print( (global_adj-distributed_adj)[:M,:N], '\n')
        print(distributed_adj[-1,:],'\n')
        print(global_adj[-1,:])
        print(np.allclose( global_adj,distributed_adj ) )
