import numpy as np
from mpi4py import MPI

from distributed_operators.gradient import distributed_gradient2d
from operators.jtv import  gradient_2d



if __name__ == '__main__' :
 # not generic enough
 #TODO adaptative number thread/ dimensions
 #! loop/freeze for high dimensions?
    M = 100
    N = 50
    global_size = np.asarray([M,N])

    comm = MPI.COMM_WORLD
   
    rank  = comm.Get_rank()
    grid_dims = np.asarray( MPI.Compute_dims(comm.Get_size(), 2) ,dtype = int )

    gradient_handler = distributed_gradient2d( global_size, grid_dims)


    cart_comm = comm.Create_cart(dims = grid_dims)


    X = np.zeros([M,N])

    if rank == 0 :
        X= np.random.rand(M,N)
    
    cart_comm.Bcast([X, MPI.DOUBLE], root=0)
    
    slice_0 = gradient_handler.cart_comm.cartslicer._get_slice_global_buffer_to_tile()[0]
    slice_1 = gradient_handler.cart_comm.cartslicer._get_slice_global_buffer_to_tile()[1]

    slices_0_start = comm.gather(slice_0.start, 0)
    slices_0_end = comm.gather(slice_0.stop, 0)

    slices_1_start = comm.gather(slice_1.start, 0)
    slices_1_end = comm.gather(slice_1.stop, 0)

    local = np.zeros(gradient_handler.cart_comm.cartslicer.tile_size)
    local = X[slice_0, slice_1]


    chunk_grad = gradient_handler.compute_grad(local)

    global_grad = np.zeros( shape= [2,M,N])
    distributed_grad = np.zeros( shape= [2,M,N])

    if rank == 0 :
        global_grad = gradient_2d(X)


    #comm.Send([chunk_grad, MPI.DOUBLE], 0)
    comm.send(chunk_grad,dest=0)
    if rank == 0 :
        for i in range( comm.Get_size() ):
            #comm.Recv([chunk_grad,MPI.DOUBLE], i )
            chunk_grad=comm.recv(source=i)

            distributed_grad[:, slices_0_start[i]:slices_0_end[i] , slices_1_start[i]:slices_1_end[i] ] = chunk_grad.copy()

        #print( (global_grad[0]-distributed_grad[0])[:M,:N], '\n')
        #print( (global_grad[1]-distributed_grad[1])[:M,:N], '\n'))
        #print(distributed_grad)
        print(np.allclose( global_grad,distributed_grad ) )

    