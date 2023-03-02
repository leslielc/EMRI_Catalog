import numpy as np
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

##======================Likelihood and Posterior (change this)=====================

def llike(params):
    #likelihood: (s-h|s-h)

    # p0_val = params[0]
    # M_val, mu_val, a_val, e0_val, Y0_val, D_val = M, mu, a, e0, Y0, dist

    M_val = params[0]
    mu_val = params[1]
    a_val = a #params[2]            # This works fine! 
    p0_val = p0 #params[3]           
    e0_val = e0 #params[4]
    Y0_val = Y0 #params[5]
    D_val = dist#params[6]
    Phi_phi0_val = Phi_phi0#params[7]
    Phi_theta0_val = Phi_theta0#params[8]
    Phi_r0_val = Phi_r0#params[9]

    waveform = AAK_waveform_model(M_val, mu_val, a_val, p0_val, e0_val, 
                                  Y0_val, D_val, qS, phiS, qK, phiK,
                                    Phi_phi0=Phi_phi0_val, Phi_theta0=Phi_theta0_val, Phi_r0=Phi_r0_val, 
                                    mich=mich, dt=delta_t, T=T)  # Generate h_plus and h_cross



    h_p_prop = waveform.real
    h_p_prop*=window
    h_p_prop_pad = zero_pad(h_p_prop)

    hp_fft_prop = np.fft.rfft(h_p_prop_pad)

    diff_f = (hp_fft - hp_fft_prop)

    inn_prod = inner_prod(diff_f,diff_f,N_t,delta_t,PSD)
    return(-0.5 * inn_prod)

def lpost(params):
    '''
    Compute log posterior
    '''

    return(llike(params)) #+ lprior(params) + )

##==========================Waveform Settings========================
delta_t = 50  # Sampling interval
# delta_t = 10 # plotting

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

####=======================True waveform==========================

waveform = AAK_waveform_model(M, mu, a, p0, e0, Y0, dist, 
                                qS, phiS, qK, phiK,
                                Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, 
                                mich=False, dt=delta_t, T=T)  # Generate h_plus and h_cross


hp = waveform.real  # Extract polarisations
hc = waveform.imag


t = np.arange(0,len(hp)*delta_t,delta_t)

window = tukey(len(hp),0.1)
hp *= window

hp_pad = zero_pad(hp)
N_t = len(hp_pad)
hp_fft = np.fft.rfft(hp_pad)
freq = np.fft.rfftfreq(N_t,delta_t)
freq[0] = freq[1]   # To "retain" the zeroth frequency
PSD = get_sensitivity(freq, sens_fn = sens_fn)
SNR2_TD = inner_prod(hp_fft,hp_fft,N_t,delta_t,PSD)

print("SNR with no gaps", np.sqrt(SNR2_TD))
print("Final time",delta_t * len(hp)/60/60/24)

##=====================Noise Setting: Currently 0=====================
variance_noise = N_t * PSD / (4*delta_t)
noise_f_real = np.random.normal(0,np.sqrt(variance_noise))
noise_f_imag = np.random.normal(0,np.sqrt(variance_noise))

noise_f = noise_f_real + 1j * noise_f_imag

data_f = hp_fft + 0*noise_f   # define the data

##===========================MCMC Settings (change this)============================
iterations = 5000 #10000  # The number of steps to run of each walker
burnin = 0
nwalkers = 10  #50 #members of the ensemble, like number of chains 森林中的小人

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

start = np.hstack((start_M, start_mu)) ##, start_a, start_p0,start_e0, start_Y0))#, start_D, start_Phi_Phi0, start_Phi_theta0, start_Phi_r0))  #nwalker x n_parameter #np.hstack((start_a))
if np.size(start.shape) == 1:
    start = start.reshape(start.shape[-1], 1)
    ndim = 1
else:
    ndim = start.shape[-1]
# breakpoint()
# The shape of the above is 32 rows with 5 columns. The number of columns is the dimension of parameter space
# and the 32 rows correspond to starting points for each walker in the emsembler. 
print("Should be zero if there is no noise", llike(start[0]))


# os.chdir('/Users/oburke/Documents/Presentations/Review_Bayeisan_GWs')
# t = np.arange(0,len(hp)*delta_t,delta_t)/60/60/24
# np.random.seed(1234)
# noise_t = 5*1e-22 * np.random.normal(0,1,len(hp))

# plt.plot(t,noise_t,color = 'green', label = 'Noise')
# plt.plot(t,hp, color = 'red', alpha = 0.5, label = 'EMRI signal')
# plt.legend(fontsize = 15)
# plt.xlim([t[-1] - 0.4,t[-1]])
# plt.ylim([-2e-21,2e-21])
# plt.xlabel(r'Time [days]',fontsize = 15)
# plt.ylabel(r'Strain',fontsize = 15)
# plt.title(r'Extreme mass-ratio inspiral',fontsize = 15)
# plt.savefig("EMRI_plot_noise.pdf")
# plt.show()

# breakpoint()

os.chdir('./mcmc_result')
moves_stretch = emcee.moves.StretchMove(a=2)  #怎么取下一个点 a:小人的步长
fp = "M_mu_10000.h5" 
backend = emcee.backends.HDFBackend(fp)
# start = backend.get_last_sample() #Continue
backend.reset(nwalkers, ndim) #Start New


# pool = get_context("fork").Pool(8)                      # Magical command to allow for multiprocessing on this nightmare M1 laptop
pool = Pool(8)                    

sampler = emcee.EnsembleSampler(nwalkers, ndim, lpost, pool = pool,
                                backend=backend, moves = moves_stretch)

sampler.run_mcmc(start,iterations, progress = True, tune=True)

# breakpoint()