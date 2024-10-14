import numpy as np
import sys

if __name__ == '__main__' :
    rng = np.random.default_rng(1234)
    new_state_state = rng.bit_generator.__getstate__()[0]["state"]["state"]
    new_state_inc = rng.bit_generator.__getstate__()[0]["state"]["inc"]

    a = rng.standard_normal(10)

    loaded_state_state = np.array( bytearray( new_state_state.to_bytes(32, sys.byteorder) ) )
    loaded_state_inc = np.array( bytearray( new_state_inc.to_bytes(32, sys.byteorder) ) )                        

    rng2 = np.random.default_rng(5678)
    new_state = rng2.bit_generator.__getstate__()
    new_state[0]["state"]["state"] = int.from_bytes(loaded_state_state, sys.byteorder)
    new_state[0]["state"]["inc"] = int.from_bytes(loaded_state_inc, sys.byteorder)

    rng2.bit_generator.__setstate__(new_state)

    b = rng2.standard_normal(10)
    
    #print(a,'\n', b)
    #print(loaded_state_state)
    print(np.allclose(a,b))