import cupy as cp
import torch

if __name__ == '__main__' :

    rng = torch.Generator(device='cuda').manual_seed(1234)

    shape = [512,512]
    n = 10

    A = cp.zeros(shape)

    for i in range(n):
        A = cp.asarray( torch.normal( torch.as_tensor(cp.zeros_like(A), device='cuda'), torch.as_tensor(cp.ones_like(A), device='cuda'),generator=rng) )

    rng2 = torch.Generator(device='cuda').manual_seed(1234)
    rng_state = rng.get_state()
    offset = rng.get_offset()
    #rng2.set_state(rng_state)
    #  https://pytorch.org/docs/stable/generated/torch.get_rng_state.html
    # get_state should not work on GPU generators
    rng2.set_offset(offset)
    B = cp.zeros(shape)


    for i in range(10):
        A = cp.asarray( torch.normal( torch.as_tensor(cp.zeros_like(A), device='cuda'), torch.as_tensor(cp.ones_like(A), device='cuda'),generator=rng) )
        B = cp.asarray( torch.normal( torch.as_tensor(cp.zeros_like(B), device='cuda'), torch.as_tensor(cp.ones_like(B), device='cuda'),generator=rng2) )

        print( cp.allclose(A,B))