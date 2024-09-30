import numpy as np

from mpi4py import MPI

from distributed_operators.gradient import distributed_gradient2d
from operators.jtv import  gradient_2d_adjoint


if __name__ == '__main__' :

    M = 12
    N = 12
    global_size = np.asarray([M,N])

    comm = MPI.COMM_WORLD
   
    rank  = comm.Get_rank()
    grid_dims = np.asarray( MPI.Compute_dims(comm.Get_size(), 2) ,dtype = int )

    gradient_handler = distributed_gradient2d( global_size, grid_dims)

    m,n = gradient_handler.cart_comm.cartslicer.tile_size


    cart_comm = comm.Create_cart(dims = grid_dims)

    X = np.zeros([2,M,N])
    local = np.zeros([2,m+1,n+1])
    local_v = np.zeros(gradient_handler.adj_cart_comm_h.cartslicer.tile_size)
    local_h = np.zeros(gradient_handler.adj_cart_comm_v.cartslicer.tile_size)

    if rank == 0 :
        X= np.random.rand(2,M,N)
        #X = np.ones([2,M,N])
        #X[0,:,:] = np.zeros([M,N])
        #X[1,:,:] = np.zeros([M,N])
        
    
    cart_comm.Bcast([X, MPI.DOUBLE], root=0)

    grid_coords = cart_comm.Get_coords(rank)    

    first_x = ( 0 == grid_coords[0])
    first_y = ( 0 == grid_coords[1])

    last_x = (grid_dims[0]-1 == grid_coords[0])
    last_y = (grid_dims[1]-1 == grid_coords[1])



    x = gradient_handler.cart_comm.cartslicer.tile_range[0][0]
    y = gradient_handler.cart_comm.cartslicer.tile_range[1][0]
   
    local_h[-m:,-n:] = X[0][x:x+m,y:y+n]
    local_v[-m:,-n:] = X[1][x:x+m,y:y+n]
   

    chunk_adj = np.zeros(gradient_handler.adj_cart_comm_h.cartslicer.tile_size)
    

    gradient_handler.compute_adjoint(chunk_adj, local_h, local_v)
    
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

        #print( (global_adj-distributed_adj)[:M,:N], '\n')
        #print(distributed_adj,'\n')
        #print(global_adj)
        print(np.allclose( global_adj,distributed_adj ) )