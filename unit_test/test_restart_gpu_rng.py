import cupy as cp
import torch
from numpy import zeros, all

import pytest

def test_restart_gpu_rng():

    rng = torch.Generator(device='cuda').manual_seed(1234)

    shape = [512,512]
    n = 10

    A = cp.zeros(shape)

    for i in range(n):
        A = cp.asarray( torch.normal( torch.as_tensor(cp.zeros_like(A), device='cuda'), torch.as_tensor(cp.ones_like(A), device='cuda'),generator=rng) )

    rng2 = torch.Generator(device='cuda').manual_seed(1234)

    offset = rng.get_offset()
    rng2.set_offset(offset)
    B = cp.zeros(shape)

    check = zeros(10)

    for i in range(10):
        A = cp.asarray( torch.normal( torch.as_tensor(cp.zeros_like(A), device='cuda'), torch.as_tensor(cp.ones_like(A), device='cuda'),generator=rng) )
        B = cp.asarray( torch.normal( torch.as_tensor(cp.zeros_like(B), device='cuda'), torch.as_tensor(cp.ones_like(B), device='cuda'),generator=rng2) )

        check[i] = cp.allclose(A,B)

    assert all(check)