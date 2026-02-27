import pytest
import numpy as np
from mpi4py import MPI

import mcmc.backend as backend_module


@pytest.fixture
def seed():
    return 1234


@pytest.fixture
def dims():
    return np.array([100, 75], "i")


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


@pytest.mark.env("serial-cpu")
@pytest.mark.env("serial-gpu")
def test_basic_check(dims, cmdopt):
    if cmdopt == "serial-gpu":
        backend_module.set_backend("cupy")
    from mcmc.backend import xp
    from mcmc.operators.gradient import Gradient2d

    X = xp.ones(dims)
    gradient_operator = Gradient2d(dims)

    grad = gradient_operator.forward(X)

    assert xp.amax(np.abs(grad)) == 0


@pytest.mark.env("serial-cpu")
@pytest.mark.env("serial-gpu")
def test_adjoint(seed, dims, cmdopt):
    if cmdopt == "serial-gpu":
        backend_module.set_backend("cupy")
    from mcmc.backend import xp
    from mcmc.operators.gradient import Gradient2d

    rng = xp.random.default_rng(seed)
    X = rng.standard_normal(dims)
    Y = rng.standard_normal(np.asarray((2, *dims)))

    gradient_operator = Gradient2d(dims)

    Hx = gradient_operator.forward(X)
    Hy = gradient_operator.adjoint(Y)

    xHy = xp.sum(X * Hy)
    Hxy = xp.sum(Hx * Y)

    assert isinstance(Hx, xp.ndarray)
    assert isinstance(Hy, xp.ndarray)
    assert xp.isclose(Hxy, xHy, atol=1e-15)


@pytest.mark.env("mpi-cpu")
@pytest.mark.env("mpi-gpu")
def test_adjoint_mpi(comm, dims, seed, cmdopt):
    rank = comm.Get_rank()
    grid_dims = np.asarray(MPI.Compute_dims(comm.Get_size(), 2), dtype=int)
    cart_comm = comm.Create_cart(dims=grid_dims)

    if cmdopt == "mpi-gpu":
        backend_module.set_backend("cupy")
        backend_module.enable_multi_gpu()
    else:
        backend_module.set_backend("numpy")

    from mcmc.backend import xp
    from mcmc.operators.mpi_gradient import MpiGradient2d

    gradient_handler = MpiGradient2d(dims, grid_dims, comm)

    rng = np.random.default_rng(seed)

    X = xp.zeros(dims)
    Y = xp.zeros((2, *dims))

    if rank == 0:
        X = rng.standard_normal(dims)
        Y = rng.standard_normal((2, *dims))

    cart_comm.Bcast([X, MPI.DOUBLE], root=0)
    cart_comm.Bcast([Y, MPI.DOUBLE], root=0)

    local_slice = (
        gradient_handler.cart_comm.cartslicer._get_slice_global_buffer_to_tile()
    )

    local_X = X[local_slice]
    local_Y = np.zeros((2, *gradient_handler.adj_cart_comm_h.cartslicer.tile_size))
    local_adj = np.zeros(gradient_handler.adj_cart_comm_h.cartslicer.tile_size)
    local_slice_h = (
        gradient_handler.adj_cart_comm_h.cartslicer._get_slice_global_buffer_to_tile()
    )
    local_slice_v = (
        gradient_handler.adj_cart_comm_v.cartslicer._get_slice_global_buffer_to_tile()
    )

    slice_h = np.s_[0, *local_slice_h]
    slice_v = np.s_[1, *local_slice_v]
    local_Y[0, ...] = Y[slice_h]
    local_Y[1, ...] = Y[slice_v]

    local_grad = gradient_handler.forward(local_X)
    local_adj = gradient_handler.adjoint(local_Y)

    assert isinstance(local_adj, xp.ndarray)
    assert isinstance(local_grad, xp.ndarray)

    local_Hxy = xp.sum(
        local_grad[0] * local_Y[0, ...] + local_grad[1] * local_Y[1, ...]
    )
    local_xHy = xp.sum(X[local_slice] * local_adj)

    Hxy = 0
    xHy = 0
    Hxy = comm.reduce(local_Hxy, MPI.SUM, root=0)
    xHy = comm.reduce(local_xHy, MPI.SUM, root=0)
    if rank == 0:
        assert np.isclose(Hxy, xHy, atol=1e-10)


if __name__ == "__main__":
    default_seed = 1234
    default_dims = np.asarray([100, 75])

    test_basic_check(default_dims)
    test_adjoint(default_seed, default_dims)

# mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m mpi4py -m pytest test_gradient.py -C mpi-gpu
