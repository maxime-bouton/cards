from mpi4py import MPI

from mcmc.TransitionKernel.TransitionKernel import PSGLA
from mcmc.models.DistributedGaussianInpainting import DistributedGaussianInpaintingModel
from mcmc.sampler.DistributedSampler import DistributedSampler

from mcmc.slicer.cartesian_comm_slicer import CartesianCommSlicer

import h5py
import json
import numpy as np


if __name__ == "__main__":
    # config_path = "/home/stephane/dev/python-mcmc/example/inpainting/config_debug.json"
    config_file = open("config.json")
    # config_file = open("config_peppers.json")
    # config_file = open(config_path)
    params = json.load(config_file)

    nb_batches = params["nbCheckpoint"]
    batch_size = params["sampleSize"] // nb_batches
    save_path = params["savePath"]
    restart_save_path = params["reloadSavePath"]
    num_batch = params["numLoadedBatch"]

    split_coeff = params["alpha"]
    reg_coeff = params["regularizationCoefficient"]
    seed = params["seed"]  #! root_seed

    data_path = params["dataPath"]

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    grid_size = np.asarray(MPI.Compute_dims(MPI.COMM_WORLD.Get_size(), 2), dtype=int)
    mpi_cart_comm = MPI.COMM_WORLD.Create_cart(grid_size)
    ranknd = np.asarray(mpi_cart_comm.Get_coords(rank))

    # data_path ="/home/stephane/dev/python-mcmc/data/inpainting_data_cameraman_ds1_isnr40.h5"

    with h5py.File(data_path, "r", driver="mpio", comm=MPI.COMM_WORLD) as data_file:
        mask = data_file["mask01"][:]
        # mask = data_file["mask"][:]
        # sigma2 = data_file["sig2"][:]
        sigma2 = data_file["sig2"][()]
        observations = data_file["data"][:]
        img_size = observations.shape

    slicer = CartesianCommSlicer(
        ranknd, grid_size, observations.shape, np.asarray([0, 0]), np.asarray([0, 0])
    )
    tile_size = slicer.tile_size
    slice = slicer._get_slice_global_buffer_to_tile()
    mask = mask[slice]
    observations = observations[slice]

    step_size_X = 0.99 * 1.0 / (8.0 / split_coeff + 1.0 / sigma2)
    X = PSGLA(tile_size, step_size_X)

    step_size_Z = 0.99 / split_coeff
    Z = PSGLA((2, *tile_size), step_size_Z)

    model = DistributedGaussianInpaintingModel(
        img_size, grid_size, observations, mask, X, Z, sigma2, reg_coeff, split_coeff
    )

    sampler = DistributedSampler(
        batch_size, nb_batches, seed, "sample", save_path, model
    )

    # load_path = "../../produced_data/sample/sample"+ str(num_batch-1)+".h5"
    # load_path = "/home/stephane/dev/python-mcmc/produced_data/sample/sample"+ str(num_batch-1)+".h5"
    # sampler.restart(load_path, restart_save_path, num_batch) #? +1
    # sampler.restart("../../produced_data/sample/sample5.h5", restart_save_path, 1)

    sampler.sample()
