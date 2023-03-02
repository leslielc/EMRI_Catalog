import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tukey
import time
import os

from few.waveform import Pn5AAKWaveform


delta_t = 10  # Sampling interval
np.random.seed(1234)    # Set the seed

use_gpu = True 
# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": True,
}

# Construct the AAK model with 5PN trajectories
start = time.time()
AAK_waveform_model = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=False)

M = 1e6; mu = 10; a = 0.9; p0 = 14; e0 = 0.7; iota0 = 1; Y0 = np.cos(iota0); dist = 1
Phi_phi0 = 2; Phi_theta0 = 3; Phi_r0 = 1.5
qS = 0.5; phiS = 2; qK = 0.8; phiK = 1;  mich = False; T = 2
start = time.time()
waveform = AAK_waveform_model(M, mu, a, p0, e0, Y0, dist, 
                                qS, phiS, qK, phiK,
                                Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, 
                                mich=False, dt=delta_t, T=T)  # Generate h_plus and h_cross
end = time.time()

print("Time taken to evaluate waveform is ", end - start, " seconds")

