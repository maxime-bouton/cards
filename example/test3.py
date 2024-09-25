import numpy as np

from mpi4py import MPI

from slicer.comm_slicer import CommSlicer
from slicer.cartesian_comm_slicer import CartesianCommSlicer
from communicator.sync_cartesian_communicator import SyncCartesianCommunicator
from operators.jtv import  gradient_2d_adjoint
from distributed_operators.gradient import chunk_gradient_2d_adjoint



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
        X = np.ones([2,M,N])
        X[0,:,:] = np.zeros([M,N])
    
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
            #print(i, coord, cartesian_comm_rows.ranknd ) #! ranknd form root thread, return [0,0]
            x = coord[0]*m
            y = coord[1]*n
            distributed_adj[ x:x+m , y:y+n ] = chunk_adj[-m:,-n:].copy()

        #print( (global_adj-distributed_adj)[:M,:N], '\n')
        print(distributed_adj,'\n')
        #print(global_adj)
        print(np.allclose( global_adj,distributed_adj ) )
