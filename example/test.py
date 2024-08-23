import cupy as cp

if __name__ == '__main__' :
    x = cp.zeros([10,10])
    #print( cp.asnumpy(x) )

    seed = 1234
    rng = cp.random.default_rng(seed)
    shape = [2,2]
    for i in range(10):
        x = rng.standard_normal(shape)

    #print( cp.asnumpy(x) )
    n = 10
    rng2 = cp.random.default_rng(seed)
    offset = n * shape[0] * shape[1]

    state = cp.random.get_random_state()
    y = state.normal(0,1, shape, dtype = float)
    #print( cp.asnumpy(y))

    z  = cp.random.normal(0,1,shape, dtype = float)
    #print( cp.asnumpy(z))

    y = rng2.standard_normal(shape)
    x = rng.standard_normal(shape)

    #cp.cuda.curand.setGeneratorOffset(rng2, offset)

    #x = state._generate_normal( cp.cuda.curand.generateNormalDouble,10*10, dtype = float )

    #rng3 = cp.cuda.curand.createGenerator(cp.cuda.curand.CURAND_RNG_PSEUDO_XORWOW)
    rng3 = cp.cuda.curand.createGenerator(cp.cuda.curand.CURAND_RNG_PSEUDO_DEFAULT)
    cp.cuda.curand.setGeneratorOffset(rng3,offset)
    #x = cp.cuda.curand.generateNormalDouble(rng3,x, 10*10, cp.zeros(shape), cp.ones(shape))

    offset = (n-1)*shape[0]*shape[1]/2
    state = cp.random.RandomState(seed = 1234, method  = cp.cuda.curand.CURAND_RNG_PSEUDO_DEFAULT)
    cp.cuda.curand.setGeneratorOffset(state._generator, offset)
    x = state.standard_normal(shape)

    state2 = cp.random.RandomState(seed = 1234, method  = cp.cuda.curand.CURAND_RNG_PSEUDO_DEFAULT)

    for i in range(n):
        y = state2.standard_normal(shape)
        print(cp.asnumpy(y))
        print('\n')

    print(cp.asnumpy(x))
    

    #for i in range(10):
        #x = rngXOR.standard_normal(shape)
        #x = cp.cuda.curand.generateNormalDouble(rngXOR, x, 10*10, cp.zeros(shape), cp.ones(shape))

