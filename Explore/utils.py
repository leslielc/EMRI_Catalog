# This is a basic function that will be used to build the year long window function

from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey

def zero_pad(data):
    """
    This function takes in a vector and zero pads it so it is a power of two.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data,(0,int((2**pow_2)-N)),'constant')

def inner_prod(sig1_f,sig2_f,N_t,delta_t,PSD):

    prefac = 4*delta_t / N_t
    sig2_f_conj = np.conjugate(sig2_f)

    return prefac * np.real(sum((sig1_f * sig2_f_conj)/PSD))




