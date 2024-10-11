import numpy as np

if __name__ == '__main__' :
    rng = np.random.default_rng(1234)
    new_state_state = rng.bit_generator.__getstate__()[0]["state"]["state"]
    new_state_inc = rng.bit_generator.__getstate__()[0]["state"]["inc"]

    a = rng.standard_normal(10)

    rng2 = np.random.default_rng(5678)
    new_state = rng2.bit_generator.__getstate__()
    new_state[0]["state"]["state"] = new_state_state
    new_state[0]["state"]["inc"] = new_state_inc

    rng2.bit_generator.__setstate__(new_state)

    b = rng2.standard_normal(10)
    
    print(np.allclose(a,b))