from mpi4py import MPI

from TransitionKernel.TransitionKernel import PSGLA
from models.DistributedGaussianInpainting import DistributedGaussianInpaintingModel
from sampler.DistributedSampler import DistributedSampler

from slicer.cartesian_comm_slicer import CartesianCommSlicer

import h5py
import json
import numpy as np


if __name__ == '__main__' :
    config_file = open("config.json")
    params = json.load(config_file)

    nb_batches = params["nbCheckpoint"]
    batch_size = params["sampleSize"] // nb_batches
    save_path = params["savePath"]
    restart_save_path = params["reloadSavePath"]

    split_coeff = params["alpha"]
    reg_coeff = params["regularizationCoefficient"]
    seed = params["seed"]

    data_path = params["dataPath"]

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    grid_size = np.asarray( MPI.Compute_dims(MPI.COMM_WORLD.Get_size(), 2) ,dtype = int )
    mpi_cart_comm = MPI.COMM_WORLD.Create_cart( grid_size )
    ranknd = np.asarray(mpi_cart_comm.Get_coords(rank) )

    with h5py.File(data_path,'r',driver='mpio', comm=MPI.COMM_WORLD) as data_file:
        mask = data_file["mask01"][:]
        sigma2 = data_file["sig2"][:]
        observations = data_file["data"][:]
        img_size = observations.shape

    slicer = CartesianCommSlicer(ranknd, grid_size, observations.shape, np.asarray([0,0]), np.asarray([0,0]))
    tile_size = slicer.tile_size
    slice = slicer._get_slice_global_buffer_to_tile()
    mask = mask[ slice]
    observations = observations[slice]

    step_size_X = 0.99 * 1./( 8./split_coeff + 1./sigma2 )
    #X = PSGLA(observations.shape, step_size_X) #! size must be changed to tile_size
    X = PSGLA( tile_size, step_size_X) 

    step_size_Z = 0.99 / split_coeff
    #Z = PSGLA( (2,*X.current_state.shape), step_size_Z) #! size must be changed to tile_size
    Z = PSGLA( (2,*tile_size), step_size_Z)
    
    model = DistributedGaussianInpaintingModel(
                img_size,
                grid_size,
                observations ,
                mask,
                X ,
                Z ,
                sigma2,
                reg_coeff,
                split_coeff)
    # conditionnals are set in the constructor

    sampler = DistributedSampler(
        batch_size,
        nb_batches,
        seed,
        "sample",
        save_path,
        model
    )
    
    sampler.sample()