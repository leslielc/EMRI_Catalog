import numpy as np
import cupy as cp
from IPython import embed

import matplotlib.pyplot as plt
from scipy.signal import tukey
import multiprocessing as mp
from multiprocessing import Pool,get_context
from lisatools.sensitivity import get_sensitivity
import os
from utils import (zero_pad, inner_prod)
from mcmc_func import (lprior)
from settings import (M, mu, a, p0, e0, iota0, Y0, 
                      dist, Phi_phi0, Phi_theta0, Phi_r0, qS, phiS, qK, phiK, 
                      mich, T, 
                      sens_fn)
from few.waveform import Pn5AAKWaveform
import emcee
np.random.seed(1234)

def zero_pad(data):
    N = len(data)
    pow_2 = xp.ceil(np.log2(N))
    return xp.pad(data,(0,int((2**pow_2)-N)),'constant')

def inner_prod(sig1_f,sig2_f,N_t,delta_t,PSD):
    prefac = 4*delta_t / N_t
    sig2_f_conj = xp.conjugate(sig2_f)
    return prefac * xp.real(xp.sum((sig1_f * sig2_f_conj)/PSD))


##======================Likelihood and Posterior (change this)=====================

def llike(params):
    #likelihood: (s-h|s-h)

    M_val = float(params[0])
    mu_val = float(params[1])
    a_val = float(params[2])            # This works fine! 
    p0_val = float(params[3])           
    e0_val = float(params[4])
    Y0_val = float(params[5])
    D_val = float(params[6])
    Phi_phi0_val = float(params[7])
    Phi_theta0_val = float(params[8])
    Phi_r0_val = float(params[9])

    waveform = AAK_waveform_model(M_val, mu_val, a_val, p0_val, e0_val, 
                                  Y0_val, D_val, qS, phiS, qK, phiK,
                                    Phi_phi0=Phi_phi0_val, Phi_theta0=Phi_theta0_val, Phi_r0=Phi_r0_val, 
                                    mich=mich, dt=delta_t, T=T)  # Generate h_plus and h_cross



    h_p_prop = waveform.real
    h_p_prop *= window
    h_p_prop_pad = zero_pad(h_p_prop)

    hp_fft_prop = xp.fft.rfft(h_p_prop_pad)

    diff_f = (hp_fft - hp_fft_prop)
    
    inn_prod = inner_prod(diff_f,diff_f,N_t,delta_t,PSD)
    return(-0.5 * inn_prod)

def lpost(params):
    '''
    Compute log posterior
    '''

    return(llike(params)) #+ lprior(params) + )

##==========================Waveform Settings========================
delta_t = 10  # Sampling interval

use_gpu = True 

if use_gpu:
    xp = cp
else:
    xp = np


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
AAK_waveform_model = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=True)

####=======================True waveform==========================

waveform = AAK_waveform_model(M, mu, a, p0, e0, Y0, dist, 
                                qS, phiS, qK, phiK,
                                Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, 
                                mich=False, dt=delta_t, T=T)  # Generate h_plus and h_cross


hp = waveform.real  # Extract polarisations
hc = waveform.imag

t = xp.arange(0,len(hp)*delta_t,delta_t)

window = xp.asarray(tukey(len(hp),0.1))
hp = window*hp

hp_pad = zero_pad(hp)
N_t = len(hp_pad)
hp_fft = xp.fft.rfft(hp_pad)
freq = xp.fft.rfftfreq(N_t,delta_t)
freq[0] = freq[1]   # To "retain" the zeroth frequency
PSD = get_sensitivity(freq, sens_fn = sens_fn)
SNR2_TD = inner_prod(hp_fft,hp_fft,N_t,delta_t,PSD)

print("SNR with no gaps", np.sqrt(SNR2_TD))
print("Final time",delta_t * len(hp)/60/60/24)
##=====================Noise Setting: Currently 0=====================

variance_noise = N_t * PSD / (4*delta_t)
noise_f_real = xp.random.normal(0,np.sqrt(variance_noise))
noise_f_imag = xp.random.normal(0,np.sqrt(variance_noise))

noise_f = noise_f_real + 1j * noise_f_imag

data_f = hp_fft + 0*noise_f   # define the data

##===========================MCMC Settings (change this)============================
iterations = 3000 #10000  # The number of steps to run of each walker
burnin = 0
nwalkers = 50  #50 #members of the ensemble, like number of chains

# n = 0
d = 1 

#here we should be shifting by the *relative* error! 

start_M = M*(1. + d * 1e-7 * np.random.randn(nwalkers,1))   # changed to 1e-6 careful of starting points! Before I started on secondaries... haha.
start_mu = mu*(1. + d * 1e-7 * np.random.randn(nwalkers,1))
start_a = a*(1. + d * 1e-7 * np.random.randn(nwalkers,1))
start_p0 = p0*(1. + d * 1e-8 * np.random.randn(nwalkers, 1))
start_e0 = e0*(1. + d * 1e-7 * np.random.randn(nwalkers, 1))
start_Y0 = Y0*(1. + d * 1e-7 * np.random.randn(nwalkers, 1))

start_D = dist*(1 + d * 1e-6 * np.random.randn(nwalkers,1))
start_Phi_Phi0 = Phi_phi0*(1. + d * 1e-6 * np.random.randn(nwalkers, 1))
start_Phi_theta0 = Phi_theta0*(1. + d * 1e-6 * np.random.randn(nwalkers, 1))
start_Phi_r0 = Phi_r0*(1. + d * 1e-6 * np.random.randn(nwalkers, 1))

start = np.hstack((start_M,start_mu, start_a, start_p0, start_e0, start_Y0, start_D, start_Phi_Phi0, start_Phi_theta0, start_Phi_r0))


if np.size(start.shape) == 1:
    start = start.reshape(start.shape[-1], 1)
    ndim = 1
else:
    ndim = start.shape[-1]

print("Should be zero if there is no noise", llike(start[0]))


#os.chdir('/Explore/mcmc_result')
moves_stretch = emcee.moves.StretchMove(a=2)  
fp = "all_params_10000.h5" 
backend = emcee.backends.HDFBackend(fp)
#start = backend.get_last_sample() #Continue
backend.reset(nwalkers, ndim) #Start New


#pool = Pool(mp.cpu_count())                    
pool = Pool(1)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lpost,# pool = pool,
                                backend=backend, moves = moves_stretch)

sampler.run_mcmc(start,iterations, progress = True, tune=True)

# breakpoint()
