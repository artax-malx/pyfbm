#! usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys


def fgn_cov(H,n,k,T):
    """ Covariance function of fractional Guassian noise.

    Args:
        H (float): Hurst index
        n (int): Number of sample points
        k (numpy.array): Time points to evaluate covariance at
        T (float): End point of simulated time interval [0,T]
    
    Returns:
        Covariance array (numpy.array)
    """
    delta = T/n
    term0 = np.power(delta*(k+1),2*H)
    term1 = np.power(delta*np.abs(k-1),2*H)
    term2 = np.power(delta*k,2*H)

    return 0.5*(term0 + term1 - 2*term2)

def first_line_circulant(H,n,m,T):
    """ Yields the first line of the circulant matrix."""
    k = np.arange(0,m)
    vals = fgn_cov(H,n,k,T)
    
    # Circulant indices: [0:(m/2-1), m/2,(m/2-1):1].
    indices = np.append(np.arange(0,m//2,1),np.arange(m//2,0,-1))
    
    # Get the first row: [r_0,r_1,...,r_(m-1),r_m,r_(m-1),...,r_1].
    vals = vals[indices]
    
    return vals

def find_size_circulant(n):
    """ Finds relevant size of circulant matrix as a power of 2 for given n. """
    m = 2
    while True:
        m = 2*m
        if m >= (n-1):
            break
    return m

def eigenvalues_circulant(H,n,T):
    """ Checks whether the eigenvalues of circulant matrix are all positive."""
    m = find_size_circulant(n)
    
    while True:
        m = 2*m
        c = first_line_circulant(H,n,m,T)
        eigenvals = np.fft.fft(c)
        eigenvals = eigenvals.real

        if np.all(eigenvals > 0):
            break

    return m, eigenvals

def fgn(eivals,m,n):
    """ Generates an fGn via circulant embedding."""
    ar = np.random.standard_normal(m//2 + 1)
    ai = np.random.standard_normal(m//2 + 1)
    
    ar[0] = np.sqrt(2)*ar[0]
    ar[m//2] = np.sqrt(2)*ar[m//2]
    
    ai[0] = 0
    ai[m//2] = 0
    
    indices = np.arange(0,(m//2+1),1)
    indices1 = np.arange((m//2-1),0,-1)
    
    ar = np.append(ar[indices],ar[indices1])
    aic = -1*ai
    ai = np.append(ai[indices],aic[indices1])
    
    W = ar + 1j*ai
    
    # Reconstruction of fGn.
    W = np.sqrt(eivals)*W
    fgn_path = np.fft.fft(W)
    fgn_path = (1/np.sqrt(2*m))*fgn_path
    fgn_path = fgn_path[0:n].real

    return fgn_path
    
def fbm(eivals,m,n):
    """ Samples an fBm by generating an fGn and cumulatively summing it."""
    fgn_path = fgn(eivals, m, n)
    fbm_path = np.cumsum(fgn_path)
    fbm_path = np.insert(fbm_path,0,0)
    
    return fbm_path

def get_fbm(H,n,T):
    """ Function to generate fBm sample path given a Hurst index, number of
    sample points and time interval.
    """ 
    m,eivals = eigenvalues_circulant(H,n,T)
    fbm_path = fbm(eivals,m,n)

    return fbm_path


if __name__ == "__main__":

    # Set the parameters.
    n = 2**17
    T = 1
    ticks = np.linspace(0,T,n+1)
    
    try:
        H = float(sys.argv[1])
    except:
        raise ValueError("Please provide the Hurst exponent: a floating number between zero and one.")
    
    fbm_path =  get_fbm(H,n,T)
    
    plt.figure(figsize=(15,8))
    plt.plot(ticks,fbm_path)
    plt.show()
