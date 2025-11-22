import numpy as np

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.misc_functions import get_speed_of_sound
from Tools.constants import *
from Tools.expansion_fan import *
from Tools.numerical_iterator import *

def get_mu(M):
    return np.arcsin(1/M)

def get_slope(theta, M, plus):
    mu = get_mu(M)
    if plus == True:
        return np.tan(theta+mu)
    else:
        return np.tan(theta-mu)

def get_K(theta, M, gamma, plus):
    nu = nu_func(M, gamma)
    if plus == True:
        return (theta - nu)
    else:
        return (theta + nu)
    
def get_theta_from_K(K, M, gamma, plus):
    nu = nu_func(M, gamma)
    if plus == True:
        return(K + nu)
    else:
        return(K-nu)

def get_M_from_K(K, theta, gamma, plus):
    if plus == True:
        nu = (theta-K)
        M = inverse_nu_func(nu, gamma)
    else:
        nu = (K-theta)
        M = inverse_nu_func(nu, gamma)
    return M


def intersect_char(K_plus, K_minus, gamma):
    theta = 0.5 * (K_minus + K_plus)
    nu = 0.5 * (K_minus - K_plus)
    M = inverse_nu_func(nu, gamma)
    return theta, M