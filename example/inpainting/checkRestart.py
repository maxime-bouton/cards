#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import h5py
import json
import matplotlib.pyplot as plt

pixelMin = 0
pixelMax = 255

configFile = open("config.json")
param = json.load(configFile)
configFile.close()

# relative path, may need to be changed
dataPath = param["savePath"]
restartDatapath = param["reloadSavePath"]

file_name = dataPath + "sample" + str(0) + ".h5"

file = h5py.File(file_name, "r")

MMSE = np.zeros_like(np.asarray(file["/MMSE"]))
restartMMSE = np.zeros_like(MMSE)

file.close()

M, N = np.shape(MMSE)
load_num = param["numLoadedBatch"] + 1
potential = []
restartPotential = []

nbBatch = param["nbCheckpoint"]

X = np.zeros([M, N, nbBatch - load_num])
restartX = np.zeros([M, N, nbBatch - load_num])

for i in range(load_num, nbBatch):
    # for i in range(1,5):
    file_name = dataPath + "sample" + str(i) + ".h5"
    file = h5py.File(file_name, "r")

    MMSE = MMSE + np.asarray(file["/MMSE"])
    X[:, :, i - load_num] = np.asarray(file["/X"])
    potential = np.append(potential, np.asarray(file["/potential"]))

    file.close()

for i in range(load_num, nbBatch):
    restartFileName = restartDatapath + "sample" + str(i) + ".h5"
    restartFile = h5py.File(restartFileName, "r")

    restartMMSE = restartMMSE + np.asarray(restartFile["/MMSE"])
    restartX[:, :, i - load_num] = np.asarray(restartFile["/X"])
    restartPotential = np.append(
        restartPotential, np.asarray(restartFile["/potential"])
    )

    restartFile.close()


MMSE = MMSE / (nbBatch - load_num)
restartMMSE = restartMMSE / (nbBatch - load_num)

plt.imshow(MMSE.T, cmap="gray", vmin=pixelMin, vmax=pixelMax)
plt.title("Estimator, initial run")
plt.colorbar()
plt.show()

plt.imshow(restartMMSE.T, cmap="gray", vmin=pixelMin, vmax=pixelMax)
plt.title("Estimator, resumed run")
plt.colorbar()
plt.show()


diff = np.amax(np.abs(X - restartX))
print("Max difference on all the checkpoints : ", diff)

plt.plot(restartPotential[:] - potential[:])
plt.title("Difference between to potential along the two run")
plt.show()
