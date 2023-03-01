import sys
print(sys.version)
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tukey
from lisatools.sensitivity import get_sensitivity
from utils import (zero_pad, inner_prod)

from few.waveform import Pn5AAKWaveform

##======================Waveform Settings====================
# Sampling interval
delta_t = 10

use_gpu = False # CHANGE ME TO "TRUE" PLEASE <--- GPU ACCELERATE.

# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "max_init_len": int(1e4),  # all of the trajectories will be well under len = 1000
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": True,
}

# Construct the AAK model with 5PN trajectories
AAK_waveform_model = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=False)

####=======================True waveform==========================

M = 1e6; mu = 10; a = 0.9; p0 = 10.0; e0 = 0.7; iota0 = 1; Y0 = np.cos(iota0); dist = 1
Phi_phi0 = 2; Phi_theta0 = 3; Phi_r0 = 1.5
qS = 0.5; phiS = 2; qK = 0.8; phiK = 1;  mich = False; T = 2

params = [M,mu,a,p0,e0,iota0,dist,Phi_phi0,Phi_theta0,Phi_r0]

begin = time.time()

waveform = AAK_waveform_model(M, mu, a, p0, e0, Y0, dist, 
                                qS, phiS, qK, phiK,
                                Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, 
                                mich=mich, dt=delta_t, T=T)  # Generate h_plus and h_cross

end = time.time()
print(end-begin)

hp = waveform.real  # Extract polarisations
hc = waveform.imag

window = tukey(len(hp),0)
hp *= window
hp_pad = zero_pad(hp)
N_t = len(hp_pad)
hp_fft = np.fft.rfft(hp_pad)
freq = np.fft.rfftfreq(N_t,delta_t)
freq[0] = freq[1]   # To "retain" the zeroth frequency

###==========PLOTS===========
t_plot = np.arange(0,len(hp)*delta_t,delta_t)
plt.plot(t_plot/60/60/24,hp)
plt.show()
# breakpoint()

PSD = get_sensitivity(freq, sens_fn = 'cornish_lisa_psd')
SNR2_TD = inner_prod(hp_fft,hp_fft,N_t,delta_t,PSD)

print("SNR with no gaps", np.sqrt(SNR2_TD))
print("Final time",delta_t * len(hp)/60/60/24)


