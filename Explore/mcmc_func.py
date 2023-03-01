
import numpy as np


M_low = 0.9e6
M_high = 1.1e6

mu_low = 9
mu_high = 11

a_low = 0.89
a_high = 0.91

p0_low = 7.8
p0_high = 8.2

e0_low = 0.09
e0_high = 0.11

iota0_low = 0.9
iota0_high = 1.1

D_low = 0.5
D_high = 1.5

Phi_phi0_low = -2*np.pi
Phi_phi0_high = 2*np.pi

Phi_theta0_low = -2*np.pi
Phi_theta0_high = 2*np.pi

Phi_r0_low = -2*np.pi
Phi_r0_high = 2*np.pi

def lprior_M(M,M_low, M_high):
    if M < M_low or M > M_high:
        return -np.inf
    else:
        return 0

def lprior_mu(mu,mu_low, mu_high):
    if mu < mu_low or mu > mu_high:
        return -np.inf
    else:
        return 0

def lprior_a(a,a_low, a_high):
    if a < a_low or a > a_high:
        return -np.inf
    else:
        return 0

def lprior_p0(p0,p0_low, p0_high):
    if p0 < p0_low or p0 > p0_high:
        return -np.inf
    else:
        return 0
    
def lprior_e0(e0,e0_low, e0_high):
    if e0 < e0_low or e0 > e0_high:
        return -np.inf
    else:
        return 0

def lprior_iota0(iota0,iota0_low, iota0_high):
    if iota0 < iota0_low or iota0 > iota0_high:
        return -np.inf
    else:
        return 0

def lprior_D(D,D_low, D_high):
    if D < D_low or D > D_high:
        return -np.inf
    else:
        return 0

def lprior_Phi_phi0(Phi_phi0,Phi_phi0_low, Phi_phi0_high):
    if Phi_phi0 < Phi_phi0_low or Phi_phi0 > Phi_phi0_high:
        return -np.inf
    else:
        return 0

def lprior_Phi_theta0(Phi_theta0,Phi_theta0_low, Phi_theta0_high):
    if Phi_theta0 < Phi_theta0_low or Phi_theta0 > Phi_theta0_high:
        return -np.inf
    else:
        return 0

def lprior_Phi_r0(Phi_r0,Phi_r0_low, Phi_r0_high):
    if Phi_r0 < Phi_r0_low or Phi_r0 > Phi_r0_high:
        return -np.inf
    else:
        return 0

def lprior(params):
    log_prior = (lprior_M(params[0],M_low, M_high)+ 
                lprior_mu(params[1],mu_low, mu_high) +
                lprior_a(params[2],a_low, a_high) + 
                lprior_p0(params[3],p0_low, p0_high) +
                lprior_e0(params[4],e0_low,e0_high)  + 
                lprior_iota0(params[5],iota0_low, iota0_high))
                # lprior_D(params[6],D_low, D_high) +
                # lprior_Phi_phi0(params[7],Phi_phi0_low, Phi_phi0_high) +
                # lprior_Phi_theta0(params[8],Phi_theta0_low, Phi_theta0_high) + 
                # lprior_Phi_r0(params[9],Phi_r0_low, Phi_r0_high))

    if np.isinf(log_prior):
        return -np.inf
    return log_prior
