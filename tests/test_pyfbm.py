#! usr/bin/env python3
import pytest
from pyfbm import pyfbm

def test_covariance_classic_brownian():
    n = 2**9
    k = 2**9
    H = 0.5
    T = 1

    cov = pyfbm.fgn_cov(H, n,k,T)

    assert cov == 0
    
