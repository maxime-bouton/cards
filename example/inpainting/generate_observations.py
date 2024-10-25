#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:46:55 2024

@author: stephane
"""

import numpy as np
import h5py


def snr(original, noised):
    return 10 * np.log10(
        np.linalg.norm(original) ** 2 / (np.linalg.norm(original - noised) ** 2)
    )


# %%
data_path = "../../data/"
file_name = "peppers.h5"

with h5py.File(data_path + file_name) as file:
    img = np.asarray(file["x"][:])

# %%

rng = np.random.default_rng(1234)
sig2 = 1.75
M, N = img.shape

gaussian_noise = rng.normal(0, np.sqrt(sig2), (M, N))
noised_img = img + gaussian_noise

# print(snr(img,noised_img))
loss = 0.4
mask = rng.binomial(1, 1 - loss, (M, N))
mask_indices = np.ravel_multi_index(np.where(mask > 0), (M, N))

observations = mask * noised_img

# %%

with h5py.File(data_path + "peppers_512_snr_40.h5", "w") as file:
    file["x"] = img
    file["mask01"] = mask
    file["sig2"] = sig2
    file["data_partial"] = observations.flatten()[mask_indices]
    file["data"] = observations
    file["mask"] = mask_indices
    file["N"] = np.asarray([M, N], dtype=int)
# %%
