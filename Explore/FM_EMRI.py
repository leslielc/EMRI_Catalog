import sys
print(sys.version)
from numba import jit
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tukey
import mpmath as mp
from lisatools.sensitivity import get_sensitivity
from utils import (zero_pad, inner_prod)
from settings import (M, mu, a, p0, e0, iota0, Y0, 
                      dist, Phi_phi0, Phi_theta0, Phi_r0, qS, phiS, qK, phiK, 
                      mich, T, 
                      sens_fn)

from few.waveform import Pn5AAKWaveform

##======================Fisher + Waveform Settings====================
# Sampling interval
delta_t = 20  
# delta_t = 10 # plotting

# difference accuracy, start j, end j
diff_info = [2, 1, -2]
# diff_info = [4, 2, -3]
# diff_info = [8, 4, -5]

#Higher precision or not
MP = True #False

print (f"Using delta_t {delta_t}s, derivative accuracy order {diff_info[0]}, Using MP? {MP}")

use_gpu = False

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
params = [M,mu,a,p0,e0,iota0,dist,Phi_phi0,Phi_theta0,Phi_r0]
waveform = AAK_waveform_model(M, mu, a, p0, e0, Y0, dist, 
                                qS, phiS, qK, phiK,
                                Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, 
                                mich=mich, dt=delta_t, T=T)  # Generate h_plus and h_cross

hp = waveform.real  # Extract polarisations
hc = waveform.imag

window = tukey(len(hp),0.1)
hp *= window
hp_pad = zero_pad(hp)
N_t = len(hp_pad)
hp_fft = np.fft.rfft(hp_pad)
freq = np.fft.rfftfreq(N_t,delta_t)
freq[0] = freq[1]   # To "retain" the zeroth frequency

###==========PLOTS===========
# t_plot = np.arange(0,len(hp)*delta_t,delta_t)
# plt.plot(t_plot/60/60/24,hp)
# plt.xlim([169,171])
# plt.show()
# breakpoint()
# plt.figure()
# plt.plot(t_plot/60/60/24,hp)
# plt.figure()
# plt.loglog(freq, abs(hp_fft)**2)
# plt.show()
# breakpoint()

PSD = get_sensitivity(freq, sens_fn = sens_fn)
SNR2_TD = inner_prod(hp_fft,hp_fft,N_t,delta_t,PSD)

print("SNR with no gaps", np.sqrt(SNR2_TD))
print("Final time",delta_t * len(hp)/60/60/24)



# Construct numerical derivatives

M_step = 1e-1   #1e-1 np.logspace(-0.5,0.8)
mu_step = 9e-6   # np.logspace(-0.3,1.0,5)
a_step = 1e-6
p0_step = 0.5*1e-6
e0_step = 1*1e-6
Y0_step = 1e-6
dist_step = 1e-4   #no difference from here
Phi_phi0_step = 1e-4
Phi_theta0_step = 1e-4
Phi_r0_step = 1e-4

deriv_fft_vec = []
waveform_steps = []
gamma_list = []
param_labels = ["M", "mu", "a", "p0", "e0", "Y0", "dist", "Phi_phi0", "Phi_theta0", "Phi_r0"]

N_params = np.size([param_labels])
uni_mat = np.eye(N_params)

coefficients = [-1/280,	+4/105,	-1/5,	+4/5,	-4/5,	+1/5,	-4/105,	1/280]
N_coefficients = len(coefficients)
start = time.time()


delta_space = np.logspace(-0.7,0.5,25) #For testing delta


for i in range(N_params): #[5]: #range(N_params)  
    for delta in [1]: #(delta_space):   ##Test for step length
        # uni_mat[i,i] = delta
        for j in range(diff_info[1],diff_info[2],-1):
            if j == 0:
                continue
            waveform_step = AAK_waveform_model(M + j*uni_mat[i,0]*M_step, 
                                               mu + j*uni_mat[i,1]*mu_step, 
                                               a + j*uni_mat[i,2]*a_step, 
                                               p0 + j*uni_mat[i,3]*p0_step, 
                                               e0 + j*uni_mat[i,4]*e0_step, 
                                               Y0 + j*uni_mat[i,5]*Y0_step, 
                                               dist + j*uni_mat[i,6]*dist_step, 
                                                qS, phiS, qK, phiK,
                                                Phi_phi0=(Phi_phi0 + j*uni_mat[i,7]*Phi_phi0_step), 
                                                Phi_theta0= (Phi_theta0 + j*uni_mat[i,8]*Phi_theta0_step), 
                                                Phi_r0=(Phi_r0 + j*uni_mat[i,9]*Phi_r0_step), 
                                                mich=mich, dt=delta_t, T=T)  # Generate h_plus and h_cross

            h_step = waveform_step.real  #h_plus time domain
            waveform_steps.append(h_step)
            # print(h_step[100:102])
            # check_fft_h = np.fft.rfft(zero_pad(window*h_step))
        
        delta_step = sum([uni_mat[i,0]*M_step, uni_mat[i,1]*mu_step, uni_mat[i,2]*a_step, uni_mat[i,3]*p0_step, 
                          uni_mat[i,4]*e0_step,  uni_mat[i,5]*Y0_step, uni_mat[i,6]*dist_step, 
                        #   qS, phiS, qK, phiK,
                          uni_mat[i,7]*Phi_phi0_step, uni_mat[i,8]*Phi_theta0_step, uni_mat[i,9]*Phi_r0_step])

        if diff_info[0] == 2:
            deriv_t = (waveform_steps[0] - waveform_steps[1])/(2*delta_step)
        elif diff_info[0] == 4:
            deriv_t = (-waveform_steps[0] + 8*waveform_steps[1] - 8*waveform_steps[2] + waveform_steps[3])/(12*delta_step) #2,1,-1,-2
        elif diff_info[0] == 8:
            deriv_t = sum([coefficients[K]*waveform_steps[K] for K in range(N_coefficients)])/delta_step

        # plt.semilogy(np.abs(deriv_t),label = param_labels[i]);plt.legend()  
        # plt.plot((deriv_t),label = param_labels[i]);plt.legend()   

        deriv_fft = np.fft.rfft(zero_pad(deriv_t*window))

        deriv_fft_vec.append(deriv_fft)

        gamma = inner_prod(deriv_fft,deriv_fft,N_t,delta_t,PSD)

        gamma_list.append(gamma)

        waveform_steps = []  # Overwrite

        print("Finished parameter", param_labels[i])


print(gamma_list) #FM diag terms

##==================================FM=====================================
# N_params = len(deriv_fft_vec)
gamma_FM = np.eye(N_params)
mp.dps = 300
gamma_FM_mp = mp.matrix(N_params, N_params)

for i in range(N_params):
    for j in range(i,N_params):
        if i == j:
            gamma_FM[i,j] = 0.5*inner_prod(deriv_fft_vec[i],deriv_fft_vec[j],N_t,delta_t,PSD)
            gamma_FM_mp[i,j] = 0.5*inner_prod(deriv_fft_vec[i],deriv_fft_vec[j],N_t,delta_t,PSD)
        else:
            gamma_FM[i,j] = inner_prod(deriv_fft_vec[i],deriv_fft_vec[j],N_t,delta_t,PSD)
            gamma_FM_mp[i,j] = inner_prod(deriv_fft_vec[i],deriv_fft_vec[j],N_t,delta_t,PSD)

gamma_FM = gamma_FM + gamma_FM.T


best_precisions = np.diag(gamma_FM)**-(1/2)
print("cond number: ", np.linalg.cond(gamma_FM))

#####==================INVERSE=====================
if not MP:
    gamma_FM_inv = np.linalg.inv(gamma_FM)
    ##===helpless SVD===
    # U,S,V = np.linalg.svd(gamma_FM)  
    # gamma_FM_inv = np.transpose(V) @ np.diag(1/S) @ np.transpose(U) 

####=================INVERSE MP====================
if MP:
    gamma_FM_mp = gamma_FM_mp + gamma_FM_mp.T
    # gamma_FM_mp = mp.matrix(gamma_FM)
    ##===helpless SVD===
    # U, S, V = mp.svd_r(gamma_FM_mp)
    # gamma_FM_inv_mp = V.T * mp.diag(S)**-1 * U.T
    gamma_FM_inv_mp = gamma_FM_mp**-1
    mp.nprint(max(gamma_FM_inv_mp * gamma_FM_mp-mp.eye(N_params)), 10)
    mp.nprint(max(gamma_FM_mp * gamma_FM_inv_mp-mp.eye(N_params)), 10)
    gamma_FM_inv = np.matrix(gamma_FM_inv_mp.tolist(), dtype=float)



print('<1?:',np.max(np.dot(gamma_FM_inv,gamma_FM)-np.eye(N_params)),np.max(np.dot(gamma_FM,gamma_FM_inv)-np.eye(N_params)))
#===test correlation===
for k in range (0,N_params):
    for l in range (k+1,N_params):
        CORR = np.abs(gamma_FM_inv[k,l]/np.sqrt(gamma_FM_inv[k,k])/np.sqrt(gamma_FM_inv[l,l]))
        if CORR > 0.997:
            print (param_labels[k], param_labels[l])
            print (gamma_FM_inv[k,l]/np.sqrt(gamma_FM_inv[k,k])/np.sqrt(gamma_FM_inv[l,l]))


delta_theta = np.sqrt(np.diag(gamma_FM_inv))
end = time.time()
# print(gamma_FM)
print("Time taken to compute FM ",end - start," seconds")
for i in range(N_params):
    print("Precision on parameter {0} is {1}, best is {2}".format(param_labels[i],delta_theta[i],(best_precisions[i])))

# plt.xlim([1e6,1.00001e6])
# plt.show()

    

