#from dsgs.operators.data import generate_random_mask, get_image

#from dsgs.operators.inpainting import SerialInpainting

from refactor.TransitionKernel.TransitionKernel import PSGLA

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

def nonNegativityProx(X):
    return np.maximum( X , 0  )
    #return X

def grad(X, sigma):
    return X/(2*sigma**2)

if __name__ == '__main__':
    sig2 = 1
    step = 0.99 / sig2
    m,n = 10,10
    A = PSGLA( dims= [m,n] , stepSize=step)

    A.setProx( nonNegativityProx )    
    A.setGrad( lambda x : grad(x,sig2) )

    rng = np.random.default_rng()
    #A.mcStep(rng)

    N = 10000
    chain = np.zeros( (N,*A.currentState.shape))

    for i in range(N):
        chain[i,:,:] = A.currentState.copy()
        A.mcStep(rng)

    sb.histplot( chain[ np.where(chain[:,0,0] > 0.01) ][:,0,0],  bins=50 )
    