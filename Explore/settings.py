##=========GPU setting========
use_gpu = True # CHANGE ME TO "TRUE" PLEASE <--- GPU ACCELERATE.
if use_gpu:
    import cupy as cp
    xp = cp
else:
    import numpy as np
    xp = np

##=========Parameters=========
# M = 1e6; mu = 10; 
# a = 0.6; p0 = 8; e0 = 0.2; 
# iota0 = 1; Y0 = np.cos(iota0); dist = 1
# Phi_phi0 = 2; Phi_theta0 = 3; Phi_r0 = 1.5
# qS = 0.5; phiS = 2; qK = 0.8; phiK = 1;  
# mich = False; T = 1

# Ollie
M = 1e6; mu = 10; a = 0.9; p0 = 8.1; e0 = 0.1; iota0 = 1; Y0 = xp.cos(iota0); dist = 1
Phi_phi0 = 2; Phi_theta0 = 3; Phi_r0 = 1.5
qS = 0.5; phiS = 2; qK = 0.8; phiK = 1;  mich = False; T = 1

##========Sensitivity==========
sens_fn = 'cornish_lisa_psd'  # 'lisasens'