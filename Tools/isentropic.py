import numpy as np

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.numerical_iterator import *

def inverse_P_P0 (P_ratio, gamma):
    """
    Numerically solves for Mach number given the P0/P ratio with isentropic flow.
    Inputs:
        P0/P (unitless)
        specific heat ratio gamma (unitless)
    Returns:
        Mach number
    """
    def func (M):
        return (1+ ((gamma-1)/2) * M**2) ** (gamma/(gamma-1))
    
    try:
        sln = numerical_iterator(func, 0, 100, P_ratio)
    except RuntimeError:
        sln = float('nan')  # Return NaN if no solution is found
    
    return sln

def P0_P(M, gamma):
    """
    Find stagnation pressure ratio via isentropic process
    Parameters:
        M: Mach number (unitless)
        gamma: ratio of specific heats (unitless)
    Returns:
        P0/P: stagnation pressure versus pressure ratio (unitless)
    """
    return ( 1 + (gamma-1)/2 * M**2) ** (gamma / (gamma-1))

def T0_T(M, gamma):
    P_rat = P0_P(M, gamma)
    return (P_rat) ** ((gamma-1)/gamma)

def rho0_rho(M, gamma):
    P_rat = P0_P(M, gamma)
    return (P_rat) ** (1/gamma)