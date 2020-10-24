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
    
def test_mean_fbm(eps=1e-5):
    #TODO: test different values of H.
    n = 2**18
    H = 0.5
    T = 1

    m, eivals = pyfbm.eigenvalues(H,n,T)
    fbm_series = pyfbm.fGn(eivals, m, n)
    mean = np.mean(fbm_series)

    assert np.abs(mean) <= eps
