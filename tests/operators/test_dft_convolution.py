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
def kernel_dims():
    return np.array([4, 3], "i")


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


@pytest.mark.env("serial-cpu")
@pytest.mark.env("serial-gpu")
def test_adjoint(seed, dims, kernel_dims, cmdopt):
    if cmdopt == "serial-gpu":
        backend_module.set_backend("cupy")

    from mcmc.backend import xp
    from mcmc.operators.dft_convolution import DftConvolution  # noqa: E402

    rng = xp.random.default_rng(seed)
    X = rng.standard_normal(dims)
    Y = rng.standard_normal(dims + kernel_dims - 1)
    kernel = rng.standard_normal(kernel_dims)

    convolution_handler = DftConvolution(dims, kernel, tuple(dims + kernel_dims - 1))

    Hx = convolution_handler.forward(X)
    Hy = convolution_handler.adjoint(Y)

    Hxy = xp.sum(Hx * Y)
    xHy = xp.sum(X * Hy)

    assert isinstance(Hx, xp.ndarray)
    assert np.isclose(Hxy, xHy, atol=1e-10)


@pytest.mark.env("mpi-cpu")
@pytest.mark.env("mpi-gpu")
def test_adjoint_gpu(seed, dims, kernel_dims, comm, cmdopt):
    rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if cmdopt == "serial-gpu":
        backend_module.set_backend("cupy")
        backend_module.enable_multi_gpu()

    from mcmc.backend import xp, gpu_context
    from mcmc.operators.mpi_dft_convolution import MpiDftConvolution

    nb_gpu = 0
    gpu_id = 0
    if cmdopt == "serial-gpu":
        nb_gpu = xp.cuda.runtime.getDeviceCount()
        gpu_id = rank % nb_gpu

    grid_dims = np.asarray(MPI.Compute_dims(comm_size, 2))

    convo_dims = dims + kernel_dims - np.ones_like(dims)

    with gpu_context(gpu_id):
        X = xp.zeros(dims)
        kernel = xp.zeros(kernel_dims)
        Y = xp.zeros(convo_dims)

    if rank == 0:
        rng = xp.random.default_rng(seed)
        with gpu_context(0):
            X = rng.standard_normal(dims)
            Y = rng.standard_normal(convo_dims)
            kernel = rng.standard_normal(kernel_dims)

    with gpu_context(gpu_id):
        comm.Bcast([X, MPI.DOUBLE], root=0)
        comm.Bcast([Y, MPI.DOUBLE], root=0)
        comm.Bcast([kernel, MPI.DOUBLE], root=0)

    convolution_handler = MpiDftConvolution(
        dims, kernel, comm, grid_dims, gpu_id=gpu_id
    )

    with gpu_context(gpu_id):
        local_X = X[
            convolution_handler.direct_communicator.cartslicer.slice_global_buffer_to_tile
        ]
        local_Y = Y[
            convolution_handler.adjoint_communicator.cartslicer.slice_global_buffer_to_tile
        ]

        local_Hx = convolution_handler.forward(local_X)
        local_Hy = convolution_handler.adjoint(local_Y)

        assert isinstance(local_Hx, xp.ndarray)
        assert isinstance(local_Hy, xp.ndarray)

        local_Hxy = np.sum(local_Hx * local_Y)
        local_xHy = np.sum(local_X * local_Hy)

    Hxy = 0
    xHy = 0

    Hxy = comm.reduce(local_Hxy, MPI.SUM, root=0)
    xHy = comm.reduce(local_xHy, MPI.SUM, root=0)

    if rank == 0:
        assert np.isclose(Hxy, xHy, atol=1e-10)


if __name__ == "__main__":
    default = "serial-cpu"
    default_seed = 1234
    default_dims = np.asarray([100, 75])
    default_kernel_dims = np.asarray([4, 3])

    test_adjoint(default_seed, default_dims, default_kernel_dims, default)

# mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m mpi4py -m pytest test_dft_convolution.py -C mpi-gpu
