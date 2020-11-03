# simulation.py - MC simulation of maximum of fBm for discretized range of H.

from pyfbm import pyfbm
import pandas as pd
import numpy as np

MC_RUNS = int(5e+6)

def simulation_max_fbm(sample_points):

    dt = 1/float(sample_points)
    T = 1

    hurst_exponents = np.arange(0.01,1.0,0.01)
    
    for H in hurst_exponents:
        arr = np.zeros(MC_RUNS)
        tau_arr = np.zeros(MC_RUNS)
        
        circ_size, eivals = pyfbm.eigenvalues_circulant(H, sample_points, T)
        
        for i in range(MC_RUNS):
            # Sample path of fBm.
            fbm_path = pyfbm.fbm(eivals, circ_size, sample_points)
            
            # Compute extremal values, excluding zero.
            max_fbm = fbm_path[1:].max()
            tau = np.argmax(fbm_path)
            
            # Store extremal values.
            arr[i] = max_fbm
            tau_arr[i] = tau*dt
        
        # Save it as a csv file.
        data = np.column_stack((arr,tau_arr))
        col1 = 'tau_H%s'%str(H)
        col2 = 'val_H%s'%str(H)
        
        df = pd.DataFrame(data, columns=[col1,col2])
        
        hurst_index = int(round(100*H))
        file_out = "max_fBm_H%02d.csv"%(hurst_index)
        df.to_csv(file_out,sep=',',index=False)
        print(f"Wrote results for H={H:.2f} to {file_out}")

if __name__ == "__main__":
    # Number of sample points has to be a power of 2.
    sample_points = 2**9
    
    simulation_max_fbm(sample_points)


