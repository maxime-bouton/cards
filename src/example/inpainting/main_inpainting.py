from TransitionKernel.TransitionKernel import PSGLA
from models.GaussianInpaintingModel import GaussianInpaintingModel
from sampler.SerialSampler import Sampler
from functionals.numpy.prox import prox_l21norm

import h5py
import json
import numpy as np

#from operators.jtv import gradient_2d, gradient_2d_adjoint

def prox_nonegativity(x):
    return np.maximum(x,0)

def gradient_2d_adjoint(X):

    v = np.zeros_like(X[0,:,:])
    v[0, :] = -X[1,0, :]
    v[1:-1, :] = X[1,:-2, :] - X[1,1:-1, :]  # -np.diff(uv[:-1,:],1,0)
    v[-1, :] = X[1,-2, :]
    v[:, 0] -= X[0,:, 0]
    v[:, 1:-1] += X[0, :, :-2] - X[0, :, 1:-1]  # -np.diff(uv[:,:-1],1,1)
    v[:, -1] += X[0, :, -2]
    return v


if __name__ == '__main__' :
    config_file = open("config.json")
    params = json.load(config_file)

    nb_batches = params["nbCheckpoint"]
    batch_size = params["sampleSize"] // nb_batches
    save_path = params["savePath"]

    split_coeff = params["alpha"]
    reg_coeff = params["regularizationCoefficient"]
    seed = params["seed"]

    data_path = params["dataPath"]

    with h5py.File(data_path,'r') as data_file:
        mask = data_file["mask01"][:]
        sigma2 = data_file["sig2"][:]
        observations = data_file["data"][:]

    step_size_X = 0.99 * 1./( 8./split_coeff + 1./sigma2 )
    X = PSGLA(observations.shape, step_size_X)

    step_size_Z = 0.99 / split_coeff
    Z = PSGLA( (2,*X.current_state.shape), step_size_Z)

    model = GaussianInpaintingModel(
                observations ,
                mask,
                X ,
                Z ,
                sigma2,
                reg_coeff,
                split_coeff)
    # conditionnals are set in the constructor

    sampler = Sampler(
                batch_size,
                nb_batches,
                seed,
                "sample",
                save_path,
                model)
    
    sampler.restart("./sample/sample5.h5", 6, "./resumed/")

    sampler.sample()