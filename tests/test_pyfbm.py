import pytest
from pyfbm import pyfbm
import numpy as np

def test_covariance_classic_brownian():
    n = 2**9
    k = 2**9
    H = 0.5
    T = 1

    cov = pyfbm.fgn_cov(H, n,k,T)

    assert cov == 0
    
def test_mean_fgn(eps=1e-4):
    #TODO: test different values of H.
    n = 2**17
    H = 0.5
    T = 1

    m, eivals = pyfbm.eigenvalues_circulant(H,n,T)
    sample_path = pyfbm.fgn(eivals, m, n)
    mean = np.mean(sample_path)

    assert np.abs(mean) <= eps

def test_cov_fbm(eps=1e-2):
    
    n = 2**10
    H = 0.3
    T = 1

    ticks = np.linspace(0,T,n+1)
    s = ticks[111]
    t = ticks[377]

    cov = 0.5*(t**(2*H) + s**(2*H) - (t-s)**(2*H))

    runs = int(5e+4)
    arr = np.zeros((2,runs))

    for i in range(runs):
        fbm_path = pyfbm.get_fbm(H,n,T)
        x = fbm_path[111]
        y = fbm_path[377]

        arr[0,i] = x 
        arr[1,i] = y

    sample_cov = np.cov(arr)[0,1]

    assert np.abs(sample_cov - cov) <= eps 


    

