from TransitionKernel.GpuTransitionKernel import GpuPSGLA
from models.GpuGaussianInpaintingModel import GpuGaussianInpaintingModel
from sampler.GpuSampler import GpuSampler
from estimator.GpuEstimatorBuilder import GpuMMSEBuilder

import h5py
import json
import numpy as np
import cupy as cp


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

    with h5py.File(data_path,'r') as data_file:
        mask = data_file["mask01"][:]
        sigma2 = data_file["sig2"][:]
        observations = data_file["data"][:]

    step_size_X = cp.asarray( 0.99 * 1./( 8./split_coeff + 1./sigma2 ) )
    X = GpuPSGLA(observations.shape, step_size_X)

    step_size_Z = cp.asarray( 0.99 / split_coeff )
    Z = GpuPSGLA( (2,*X.current_state.shape), step_size_Z)

    model = GpuGaussianInpaintingModel(
                cp.asarray(observations) ,
                cp.asarray(mask),
                X ,
                Z ,
                cp.asarray(sigma2),
                cp.asarray(reg_coeff),
                cp.asarray(split_coeff)
                )
    # conditionnals are set in the constructor

    mmse_handler = GpuMMSEBuilder( X.current_state.shape )

    sampler = GpuSampler(
                batch_size,
                nb_batches,
                seed,
                "sample",
                save_path,
                model)
    
    #sampler.restart("./sample/sample5.h5", 6, restart_save_path)

    sampler.sample()