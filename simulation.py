# simulation.py - MC simulation of maximum of fBm for discretized range of H.

from pyfbm import pyfbm
import pandas as pd
import numpy as np

# Number of sample points has to be a power of 2.
n = 2**9

hurst_exponents = np.arange(0.01,1.0,0.01)

dt = 1/float(n)
T = 1

# Number of sample paths.
M = int(5e+5)

for H in hurst_exponents:
    arr = np.zeros(M)
    tau_arr = np.zeros(M)
    
    m,eivals = pyfbm.eigenvalues(H,n,T)
    
    for i in range(M):
        # Values of fBm.
        vals = pyfbm.fbm(eivals,m,n)
        
        # Compute extremal values, excluding zero.
        max_fbm = vals[1:].max()
        tau = np.argmax(vals)
        
        # Store extremal values.
        arr[i] = max_fbm
        tau_arr[i] = tau*dt
    
    # Save it as a csv file.
    data = np.column_stack((arr,tau_arr))
    col1 = 'tau_H%s'%str(H)
    col2 = 'val_H%s'%str(H)
    
    df = pd.DataFrame(data, columns=[col1,col2])
    
    number = int(round(100*H))
    file_out = "max_fBm_H%02d.csv"%(number)
    df.to_csv(file_out,sep=',',index=False)
    print(f"Wrote results for H={H} to {file_out}")
