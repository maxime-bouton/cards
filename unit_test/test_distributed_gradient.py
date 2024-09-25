import numpy as np
from mpi4py import MPI

from distributed_operators.gradient import distributed_gradient2d
from operators.jtv import  gradient_2d



if __name__ == '__main__' :
 # not generic enough
 #TODO adaptative number thread/ dimensions
    M = 15
    N = 15
    global_size = np.asarray([M,N])

    comm = MPI.COMM_WORLD
   
    rank  = comm.Get_rank()
    grid_dims = np.asarray( MPI.Compute_dims(comm.Get_size(), 2) ,dtype = int )

    gradient_handler = distributed_gradient2d( global_size, grid_dims)
    m = gradient_handler.cart_comm.cartslicer.tile_size[0]
    n = gradient_handler.cart_comm.cartslicer.tile_size[1]

    #print(gradient_handler.cart_comm.cartslicer.slice_global_buffer_to_tile)

    cart_comm = comm.Create_cart(dims = grid_dims)


    X = np.zeros([M,N])

    if rank == 0 :
        X= np.random.rand(M,N)
    
    cart_comm.Bcast([X, MPI.DOUBLE], root=0)


    start_i = gradient_handler.cart_comm.cartslicer.tile_range[0][0]
    start_j = gradient_handler.cart_comm.cartslicer.tile_range[1][0]
    

    local = np.zeros(gradient_handler.cart_comm.cartslicer.tile_size)
    local = X[start_i:start_i+m, start_j: start_j+n]

    chunk_grad = gradient_handler.compute_grad(local)

    global_grad = np.zeros( shape= [2,M,N])
    distributed_grad = np.zeros( shape= [2,M,N])

    if rank == 0 :
        global_grad = gradient_2d(X)

    comm.Send([chunk_grad, MPI.DOUBLE], 0)


    comm.Send([chunk_grad, MPI.DOUBLE], 0)
    if rank == 0 :
        for i in range( comm.Get_size() ):
            comm.Recv([chunk_grad,MPI.DOUBLE], i )
            coord = cart_comm.Get_coords(i)
            #! does not work if number of threads does not divide the dimensions
            #! m,n values not accessible from root
            x = coord[0]*m
            y = coord[1]*n
            #slice = gradient_handler.cart_comm.cartslicer.slice_global_buffer_to_tile
            distributed_grad[:, x:x+m , y:y+n ] = chunk_grad[:,:m,:n].copy()

        #print( (global_grad[0]-distributed_grad[0])[:M,:N], '\n')
        #print( (global_grad[1]-distributed_grad[1])[:M,:N], '\n'))
        #print(distributed_grad)
        print(np.allclose( global_grad,distributed_grad ) )