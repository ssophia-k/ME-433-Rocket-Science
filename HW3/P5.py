"""
Try to see if we can make a turn of more than 37.07 degrees by splitting it into many shocks
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

import os, sys
from pathlib import Path
sys.path.insert(0, os.fspath(Path(__file__).parents[1]))
from Tools.constants import *
from Tools.oblique_shock import *

M_1 = 3.0
gamma = 1.3
DEG_PER_TURN = 1.0 # degrees per turn

def P0_P(M, gamma):
    """
    Find stagnation pressure ratio via isentropic process
    Parameters:
        M: Mach number (unitless)
        gamma: ratio of specific heats (unitless)
    Returns:
        P0/P: stagnation pressure versus pressure ratio (unitless)
    """
    return (1 + (gamma-1)/2 * M**2) ** (gamma/(gamma-1))

def P02_P01(M_1, gamma, theta_deg):
    """
    Find stagnation pressure ratio across one oblique shock
    Parameters:
        M_1: incoming Mach number (unitless)
        gamma: ratio of specific heats (unitless)
        theta_deg: shock angle in degrees
    Returns: 
        P02/P01: stagnation pressure ratio across oblique shock (unitless)
        M_2: outgoing Mach number (unitless) (added to help chain calculations)
    """
    beta, P2_1, T2_1, M_2 = mach_function(M_1, gamma, theta_deg)
    P01_P1 = P0_P(M_1, gamma)
    P02_P2 = P0_P(M_2, gamma)
    return P02_P2 * P2_1 / P01_P1, M_2

M_current = M_1
Tot_degrees = 0.0
while M_current > 1:
    P_ratio, M_next = P02_P01(M_current, gamma, DEG_PER_TURN)
    if np.isnan(M_next):  # stop if no attached-shock solution exists
        break
    M_current = M_next
    Tot_degrees += DEG_PER_TURN

print(f"After turning {DEG_PER_TURN} degrees for a total of {Tot_degrees} degrees, new Mach number is {M_current:.3f}")

