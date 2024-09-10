import cupy as cp

if __name__ == '__main__' :
    x = cp.zeros([10,10])
    #print( cp.asnumpy(x) )

    seed = 1234
    """
    rng = cp.random.default_rng(seed)
    
    for i in range(10):
        x = rng.standard_normal(shape)
        """

    shape = [1024,1024]
    shape = [512,512]
    shape = [128,100]
    #shape = [128,128]
    #shape = [10,10]
    n = 10

    offset = (n-1)*shape[0]*shape[1]//2
    print(offset)
    #state = cp.random.RandomState(seed = 1234, method  = cp.cuda.curand.CURAND_RNG_PSEUDO_MRG32K3A)
    state = cp.random.RandomState(seed = 1234, method  = cp.cuda.curand.CURAND_RNG_PSEUDO_DEFAULT)
    
    cp.cuda.curand.setGeneratorOffset(state._generator, offset)
    x = state.standard_normal(shape)

    #state2 = cp.random.RandomState(seed = 1234, method  = cp.cuda.curand.CURAND_RNG_PSEUDO_MRG32K3A)
    state2 = cp.random.RandomState(seed = 1234, method  = cp.cuda.curand.CURAND_RNG_PSEUDO_DEFAULT)

    for i in range(n):
        y = state2.standard_normal(shape)
        #print(cp.asnumpy(y))
        #print('\n')
        
    #print(cp.asnumpy(x))
    
    for i in range(5):
        x = state.standard_normal(shape)
        y = state2.standard_normal(shape)
        #print(cp.asarray(x-y))
        print( cp.allclose(x,y))

