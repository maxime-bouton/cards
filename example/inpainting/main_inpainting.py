from TransitionKernel.TransitionKernel import PSGLA
from models.GaussianInpaintingModel import GaussianInpaintingModel
from sampler.SerialSampler import Sampler

import h5py
import json


if __name__ == "__main__":
    config_file = open("config.json")
    # config_file = open("config_peppers.json")
    params = json.load(config_file)

    nb_batches = params["nbCheckpoint"]
    batch_size = params["sampleSize"] // nb_batches
    save_path = params["savePath"]
    restart_save_path = params["reloadSavePath"]
    num_batch = params["numLoadedBatch"]

    split_coeff = params["alpha"]
    reg_coeff = params["regularizationCoefficient"]
    seed = params["seed"]

    data_path = params["dataPath"]

    with h5py.File(data_path, "r") as data_file:
        mask = data_file["mask01"][:]
        # mask = data_file["mask"][:]
        # sigma2 = data_file["sig2"][:]
        sigma2 = data_file["sig2"][()]
        observations = data_file["data"][:]

    step_size_X = 0.99 * 1.0 / (8.0 / split_coeff + 1.0 / sigma2)
    X = PSGLA(observations.shape, step_size_X)

    step_size_Z = 0.99 / split_coeff
    Z = PSGLA((2, *X.current_state.shape), step_size_Z)

    model = GaussianInpaintingModel(
        observations, mask, X, Z, sigma2, reg_coeff, split_coeff
    )
    # conditionnals are set in the initialization

    sampler = Sampler(batch_size, nb_batches, seed, "sample", save_path, model)

    # load_path = "../../produced_data/sample/sample"+str(num_batch-1)+".h5"
    # sampler.restart(load_path, num_batch, restart_save_path)

    sampler.sample()
