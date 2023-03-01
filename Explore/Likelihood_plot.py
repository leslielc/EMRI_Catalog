from os import lseek
import sys
print(sys.version)


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tukey

from lisatools.sensitivity import get_sensitivity
from lisatools.diagnostic import (inner_product,
    fisher,
    covariance
)
from utils import (zero_pad, inner_prod)

from few.waveform import Pn5AAKWaveform


def llike(params):
    M_val = params[0]
    mu_val = params[1]
    a_val = params[2]            # This works fine! 
    p0_val = params[3]           
    e0_val = params[4]
    iota0_val = params[5]; Y0_val = np.cos(iota0_val)
    D_val = params[6]


    Phi_phi0_val = params[7]
    Phi_theta0_val = params[8]
    Phi_r0_val = params[9]

    waveform = AAK_waveform_model(M_val, mu_val, a_val, p0_val, e0_val, 
                                  Y0_val, D_val, qS, phiS, qK, phiK,
                                    Phi_phi0=Phi_phi0_val, Phi_theta0=Phi_theta0_val, Phi_r0=Phi_r0_val, 
                                    mich=False, dt=delta_t, T=T)  # Generate h_plus and h_cross



    h_p_prop = waveform.real

    h_p_prop*=window
    h_p_prop_pad = zero_pad(h_p_prop)
    # plt.plot(np.arange(0,len(hp_pad)*delta_t,delta_t), h_p_prop_pad-hp_pad, label=mu_val-10)

    hp_fft_prop = np.fft.rfft(h_p_prop_pad)

    diff_f = (hp_fft - hp_fft_prop)

    # plt.plot(np.arange(len(hp_fft)), diff_f, label=mu_val-10)

    inn_prod = inner_prod(diff_f,diff_f,N_t,delta_t,PSD)
    return(-0.5 * inn_prod)

delta_t = 50  # Sampling interval
# delta_t = 10 # plotting
np.random.seed(1234)    # Set the seed

use_gpu = False

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
AAK_waveform_model = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=False)

# M = 1e6; mu = 10; a = 0.9; p0 = 8.1; e0 = 0.1; iota0 = 1; Y0 = np.cos(iota0); dist = 1
# Phi_phi0 = 2; Phi_theta0 = 3; Phi_r0 = 1.5
# qS = 0.5; phiS = 2; qK = 0.8; phiK = 1;  mich = False; T = 1

# # Ollie
M = 1e6; mu = 10; a = 0.9; p0 = 8.1; e0 = 0.1; iota0 = 1; Y0 = np.cos(iota0); dist = 1
Phi_phi0 = 2; Phi_theta0 = 3; Phi_r0 = 1.5
qS = 0.5; phiS = 2; qK = 0.8; phiK = 1;  mich = False; T = 1


# # Chang
# M = 1e6; mu = 10; 
# a = 0.2; p0 = 8.5; e0 = 0.1; 
# iota0 = 1; Y0 = np.cos(iota0); 
# dist = 1
# Phi_phi0 = 2; Phi_theta0 = 3; Phi_r0 = 1.5
# qS = 0.5; phiS = 2; qK = 0.8; phiK = 1;  
# mich = False; T = 1


params = [M,mu,a,p0,e0,iota0,dist,Phi_phi0,Phi_theta0,Phi_r0]

waveform = AAK_waveform_model(M, mu, a, p0, e0, Y0, dist, 
                                qS, phiS, qK, phiK,
                                Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, 
                                mich=False, dt=delta_t, T=T)  # Generate h_plus and h_cross


hp = waveform.real  # Extract polarisations
hc = waveform.imag

# breakpoint()
t = np.arange(0,len(hp)*delta_t,delta_t)
plt.plot(t,hp)
plt.show()
window = tukey(len(hp),0.0)
hp *= window

hp_pad = zero_pad(hp)
N_t = len(hp_pad)
hp_fft = np.fft.rfft(hp_pad)
freq = np.fft.rfftfreq(N_t,delta_t)
freq[0] = freq[1]   # To "retain" the zeroth frequency
PSD = get_sensitivity(freq, sens_fn = 'cornish_lisa_psd')
SNR2_TD = inner_prod(hp_fft,hp_fft,N_t,delta_t,PSD)

print("SNR with no gaps", np.sqrt(SNR2_TD))
print("Final time",delta_t * len(hp)/60/60/24)

np.random.seed(1234)
# delta_Phi_r0 = 0.0824183062296476 #0.0783815396781209
# Phi_r0_range = np.arange(Phi_r0-3*delta_Phi_r0,Phi_r0+3*delta_Phi_r0, delta_Phi_r0/5)
# llike_vec = []
# for item in Phi_r0_range:
#     print(item)
#     params = [M, mu,a,p0,e0,iota0,dist,Phi_phi0,Phi_theta0,item]
#     llike_val = llike(params)
#     llike_vec.append(llike_val)
# plt.plot(Phi_r0_range,np.exp(np.array(llike_vec)))
# plt.axvline(x = Phi_r0 - delta_Phi_r0)
# plt.axvline(x = Phi_r0 + delta_Phi_r0)
# plt.xlabel(r'dist')
# plt.ylabel(r'Likelihood')
# plt.show()

delta = 1.7547218705452398e-07 # #0.0783815396781209
P0 = p0

P_range = np.arange(P0-3*delta,P0+3*delta, delta/5)
llike_vec = []

for item in P_range:
    print(item)
    params = [M,mu,a,item,e0,iota0,dist,Phi_phi0,Phi_theta0,Phi_r0] #[M, mu,a,p0,e0,iota0,dist,Phi_phi0,Phi_theta0,Phi_r0]
    llike_val = llike(params)
    llike_vec.append(llike_val)
# plt.legend()
plt.figure()
plt.plot(P_range,np.exp(np.array(llike_vec)))
plt.axvline(x = P0 - delta)
plt.axvline(x = P0 + delta)
plt.xlabel(r'dist')
plt.ylabel(r'Likelihood')
plt.show()

    

# M works
# mu works
# a works
# p0 works
# e0 works
# iota0 works
# Dist works

#Phi_phi0 works
#Phi_theta0 works
#Phi_r0 works

# Fuck.

# array([2.08950932e-02, 9.73311107e-07, 1.40092366e-07, 9.50007654e-08,
    #    2.34613180e-07, 1.27630179e-07, 1.74383304e-02, 1.02740111e-02,
    #    1.97943940e-02, 7.83815397e-02])




