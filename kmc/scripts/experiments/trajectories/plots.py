import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory_result(fname):
    with open(fname, 'r') as f:
            results = np.load(f)
            avg_accept = results["avg_accept"]
            avg_accept_est = results["avg_accept_est"]
            vols = results["vols"]
            vols_est = results["vols_est"]
            Ds = results["Ds"]
        
    avg_accept_mean = np.mean(avg_accept, 0)
    avg_accept_lower = np.percentile(avg_accept, 5, 0)
    avg_accept_upper = np.percentile(avg_accept, 95, 0)
    
    avg_accept_est_mean = np.mean(avg_accept_est, 0)
    avg_accept_est_lower = np.percentile(avg_accept_est, 5, 0)
    avg_accept_est_upper = np.percentile(avg_accept_est, 95, 0)
    
    vol_mean = np.mean(vols, 0)
    vol_lower = np.percentile(vols, 5, 0)
    vol_upper = np.percentile(vols, 95, 0)
    
    vol_est_mean = np.mean(vols_est, 0)
    vol_est_lower = np.percentile(vols_est, 5, 0)
    vol_est_upper = np.percentile(vols_est, 95, 0)
    
    plt.plot(Ds, avg_accept_mean, 'r')
    plt.plot(Ds, avg_accept_lower, 'r--')
    plt.plot(Ds, avg_accept_upper, 'r--')
    
    plt.plot(Ds, avg_accept_est_mean, 'b')
    plt.plot(Ds, avg_accept_est_lower, 'b--')
    plt.plot(Ds, avg_accept_est_upper, 'b--')
    
    plt.figure()
    plt.plot(Ds, vol_mean, 'r')
    plt.plot(Ds, vol_lower, 'r--')
    plt.plot(Ds, vol_upper, 'r--')
    
    plt.plot(Ds, vol_est_mean, 'b')
    plt.plot(Ds, vol_est_lower, 'b--')
    plt.plot(Ds, vol_est_upper, 'b--')
    plt.show()