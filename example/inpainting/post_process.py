#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%

import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# %%

# ! relative path to the configuration file, may need to be changed
#configFile = open("config.json")
configFile = open("config_peppers.json")
param = json.load(configFile)
path_origin = param["originPath"]

nbBatch = param["nbCheckpoint"]
path_origin = param["originPath"]
dataPath = param["dataPath"]
path = param["savePath"]

# %%
# original image is stored in the data file with the name "x"
with h5py.File(dataPath, "r") as file_origin:
    original = file_origin["x"][:]

# %%
pixelMin = 0.0
pixelMax = 255

plt.imshow(original.T, cmap="gray", vmin=pixelMin, vmax=pixelMax)
plt.title("Truth")
plt.colorbar()
plt.show()


# %% observations

with h5py.File(dataPath, "r") as file:
    observations = file["data"][:]
    mask = file["mask01"][:]
    #sigma = np.sqrt(file["/sig2"][:])
    sigma = np.sqrt(file["/sig2"][()])

plt.imshow(observations.T, cmap="gray", vmin=pixelMin, vmax=pixelMax)
plt.title("Observations")
plt.colorbar()
plt.show()



# %% forming the MMSE estimator and the potential
file_name = path + "sample" + str(0) + ".h5"
with h5py.File(file_name, "r") as file:
    MMSE = np.zeros(original.shape, dtype="d")
    potential = file["/potential"][:]
    time = file["/computation_time"][:]

burnin = 5
for i in range(burnin, nbBatch):
    file_name = path + "sample" + str(i) + ".h5"
    with h5py.File(file_name, "r") as file:
        MMSE = MMSE + file["/MMSE"][:]

for i in range(1, nbBatch):
    file_name = path + "sample" + str(i) + ".h5"

    with h5py.File(file_name, "r") as file:
        potential = np.append(potential, file["/potential"][:])
        time = np.append(time, file["/computation_time"][:])

MMSE = MMSE / (nbBatch - burnin)
atime = np.mean(time)
std_time = np.std(time)
total_time = np.sum(time)

# %% display estimator
plt.imshow(MMSE.T, cmap="gray", vmin=pixelMin, vmax=pixelMax)
plt.title("MMSE")
plt.colorbar()
plt.show()

plt.imshow(np.abs(original.T - MMSE.T), cmap="gray")
plt.title("Absolute error Truth/Estimator")
plt.colorbar()
plt.show()


plt.plot(potential)
plt.title("Potential")
#plt.savefig("potential")
plt.show()

# mmse = MMSE.astype(np.uint8)
# img = Image.fromarray(mmse.T)
# img.save("reconst.png")

# obs = observations.astype(np.uint8)
# img_noise = Image.fromarray(obs.T)
# img_noise.save("inpCam.png")


# %% reconstruction metrics
SNR = 10 * np.log10(
    np.linalg.norm(original) ** 2 / (np.linalg.norm(original - MMSE) ** 2)
)
SNR_ori = 10 * np.log10(
    np.linalg.norm(original) ** 2 / (np.linalg.norm(original - observations) ** 2)
)

ssim_noise = ssim(
    original, observations, data_range=np.amax(original) - np.amin(original)
)
ssim_recon = ssim(original, MMSE, data_range=np.amax(original) - np.amin(original))

# %%
print("Observations/Truth SNR : ", SNR_ori)
print("Reconstruction/Truth SNR : ", SNR)
print("Observations/Truth SSIM : ", ssim_noise)
print("Reconstruction/Truth SSIM : ", ssim_recon)

print(
    "Reconstruction total time={:.3e}, atime={:.3e}, std={:.3e}".format(
        total_time, atime, std_time
    )
)
