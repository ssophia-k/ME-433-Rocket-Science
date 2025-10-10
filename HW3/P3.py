"""
In class it was discussed that it is more “efficient” in terms of stagnation pressure to
utilize two oblique shocks to turn a flow than a single oblique shock whose turning angle
is equal to the sum of the two. Empirically demonstrate that this is indeed the case for the
case of M = 3, 𝛾 = 1.3, and total flow turning of 20 degrees. Plot the stagnation pressure
of such a setup as a function of initial step angle from 0 to 20 degrees.
"""

import numpy as np
import matplotlib.pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.constants import *
from Tools.oblique_shock import *

M_1 = 3
gamma = 1.3

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
    P02_P01 = P02_P2 * P2_1 * (1 / P01_P1)
    return P02_P01, M_2



theta_1_deg = np.linspace(10e-5, 20-10e-5, 1000) # first shock angle in degrees
relative_stagnation = []
for th1 in theta_1_deg:
    # First shock
    P02_P01_val, M2 = P02_P01(M_1, gamma, th1)

    # Second shock with remaining angle
    th2 = 20 - th1
    P03_P02_val, M3 = P02_P01(M2, gamma, th2)

    # Chain them
    P03_P01 = P02_P01_val * P03_P02_val
    relative_stagnation.append(P03_P01)

relative_stagnation = np.array(relative_stagnation)

# Calculate edge case of single shock with 20 degree deflection
P03_P01_single, _ = P02_P01(M_1, gamma, 20)

plt.plot(theta_1_deg, relative_stagnation, label="Two shocks (split turn)")
plt.scatter([theta_1_deg[0], theta_1_deg[-1]],
            [P03_P01_single, P03_P01_single],
            color='red', marker='o', zorder=5,
            label="Single Shock (20 deg)")

plt.legend()
plt.xlabel("First Shock Angle (degrees)")
plt.ylabel("Stagnation Pressure Ratio $P_{0,3}/P_{0,1}$")
plt.title("Stagnation Pressure Ratio vs First Shock Angle")
plt.grid(True)
plt.show()


