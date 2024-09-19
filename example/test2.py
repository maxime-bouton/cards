#import cupy as cp
import numpy as np

from mpi4py import MPI

from slicer.comm_slicer import CommSlicer
from slicer.cartesian_comm_slicer import CartesianCommSlicer
from communicator.sync_cartesian_communicator import SyncCartesianCommunicator
from operators.jtv import  chunk_gradient_2d_adjoint, gradient_2d


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



if __name__ == '__main__' :
    """
    ranknd = np.asarray([0,1])
    send_size = np.asarray([2,2])
    recv_size = send_size
    grid_size = np.asarray([2,2])
    global_size = np.asarray([10,10])
    slicer = CartesianCommSlicer( ranknd , grid_size, global_size , send_size, recv_size )
"""
    M = 12
    N = 12
    global_size = np.asarray([M,N])

    comm = MPI.COMM_WORLD
   
    rank  = comm.Get_rank()
    grid_dims = np.asarray( MPI.Compute_dims(comm.Get_size(), 2) ,dtype = int )
    m = M//grid_dims[0]
    n = N//grid_dims[1]

    #cart_comm = MPI.Intracomm.Create_cart(comm, dims = grid_dims)
    cart_comm = comm.Create_cart(dims = grid_dims)

    #buffer_size = np.asarray( global_size // grid_dims ,dtype = int)

    send_size = np.asarray([1,1],dtype = int) # send one column and one line
    #send_size = np.asarray([0,1],dtype = int) # send one column
    recv_size = send_size.copy()

    cartesian_comm = SyncCartesianCommunicator(comm, grid_dims, global_size, send_size, recv_size, backward=False)

    X = np.zeros([M,N])

    if rank == 0 :
        #print(cartesian_comm.ranknd)
        #print(cartesian_comm.src)
        #print(cartesian_comm.dest)

        X= np.random.rand(M,N)
    
    cart_comm.Bcast([X, MPI.DOUBLE], root=0)

    start_i = cartesian_comm.cartslicer.tile_range[0][0]
    start_j = cartesian_comm.cartslicer.tile_range[1][0]
    

    local = np.ones(cartesian_comm.cartslicer.facet_size) * rank
    #local = np.ones(cartesian_comm.cartslicer.tile_size) * rank
    local[:m,:n] = X[start_i: start_i+m, start_j: start_j+n].copy()


    cartesian_comm.update_borders(local)

    grid_coords = cart_comm.Get_coords(rank)
    
    last_x = (grid_dims[0]-1 == grid_coords[0])
    last_y = (grid_dims[1]-1 == grid_coords[1])

    chunk_grad = np.ones([2,*cartesian_comm.cartslicer.tile_size]) * rank
    #chunk_grad[:,:m,:n] = np.concatenate([local[None,...],local[None,...]])

    chunk_grad = chunk_gradient_2d(local , np.asarray([last_x,last_y]) ) 
    


    #local_grad = gradient_2d(local)
   
    #if rank == 0:
    #    print(X[4:4+m , 4:4+n])
    #if rank == 4 :
        #print(local)
        #print(grid_coords)
        #print(chunk_grad[0][:m,:n] - local_grad[0][:m,:n])
        #print(chunk_grad[1] - local_grad[1][:m,:n])
        #print(start_i, start_j)
    #print(chunk_grad[0] )
    #print(local_grad[0])
    #print(chunk_grad[1] )
    #print(local_grad[1])

    global_grad = np.zeros( shape= [2,M,N])
    distributed_grad = np.zeros( shape= [2,M,N])

    if rank == 0 :
        global_grad = gradient_2d(X)


    #print(local)
    print(chunk_grad.shape, cartesian_comm.cartslicer.tile_size, cartesian_comm.cartslicer.facet_size  ,rank)

    comm.Send([chunk_grad, MPI.DOUBLE], 0)


    comm.Send([chunk_grad, MPI.DOUBLE], int(0), int(rank))
    if rank == 0 :
        for i in range( int(comm.Get_size()) ):
            comm.Recv([chunk_grad,MPI.DOUBLE], int(i), int(i) )
            coord = cart_comm.Get_coords(i)
            x = coord[0]*m
            y = coord[1]*n
            distributed_grad[:, x:x+m , y:y+n ] = chunk_grad[:,:m,:n].copy()


        #distributed_grad[0][:,-1] = np.zeros([M])
        #distributed_grad[1][-1,:] = np.zeros([N])

        #print( (global_grad[0]-distributed_grad[0])[:M,:N], '\n')
        #print( (global_grad[1]-distributed_grad[1])[:M,:N], '\n')
        #print(global_grad[1]-distributed_grad[1])
        #print(global_grad[0][:M,:N],'\n')
        print(distributed_grad)
        #print(global_grad[1])
        print(np.allclose( global_grad,distributed_grad ) )

   

    #mpiexec -n 2 python -m mpi4py -m pytest tests/mpi/test_mpi_operators.py

    #MPI.Finalize()

    """
    mamba create --name gpu-dsgs python=3.11 numpy numba mpi4py "h5py>=2.9=mpi*" -y
    mamba activate gpu-dsgs
    mamba install scipy scikit-image matplotlib imageio tqdm jupyterlab pytest black flake8 isort coverage pre-commit sphinx sphinx_rtd_theme sphinxcontrib-bibtex sphinx-autoapi furo conda-build -y
    conda develop src
    """