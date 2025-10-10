import numpy as np

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.numerical_iterator import *

def nu_func(M, gamma):
    """
    Prandtl-Meyer function
    Parameters:
        M: incoming Mach number (unitless)
        gamma: ratio of specific heats (unitless)
    Returns:
        nu: Prandtl-Meyer function value (radians)
    """
    if M < 1:
        raise ValueError("M must be >= 1")
    
    return np.sqrt((gamma+1)/(gamma-1)) * np.arctan(np.sqrt(((gamma-1)/(gamma+1)) * (M**2-1))) - np.arctan(np.sqrt(M**2-1))

def get_M2_from_nu(M1, gamma, theta_deg):
    """
    Given M1 and delta_nu, solve for M2 using the Prandtl-Meyer function.
    Parameters:
        M1: incoming Mach number (unitless)
        gamma: ratio of specific heats (unitless)
        theta_deg: theta = nu2-nu1, change in Prandtl-Meyer function (degrees)
    Returns:
        M2: outgoing Mach number after expansion fan (unitless)
    """
    if M1 < 1:
        raise ValueError("M1 must be >= 1")
    
    nu1 = nu_func(M1, gamma)
    nu2 = nu1 + np.deg2rad(theta_deg)  # nu2 = nu1 + delta_nu

    # Define the function to find root for
    def func(M):
        return nu_func(M, gamma) - nu2

    # Initial guess for M2
    M2_guess = M1 
    # Use numerical iterator to find root
    # def numerical_iterator(func, start, end, goal_y, tol=1e-6, max_iter=1000):
    M2 = numerical_iterator(func, M2_guess, 100, 0, tol=1e-6, max_iter=1000) # end=100 since thats wayyy too high

    return M2