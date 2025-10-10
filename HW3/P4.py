"""
Extend problem 3 from two turns to three turns and plot a 2D surface showing stagnation
pressure as a function of the first two step angles.
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
TOTAL_TURN = 20.0 # degrees
TOL = 1e-4 # numerical tolerance

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

def P04_P01_three(M1, gamma, th1, th2, total_deg=TOTAL_TURN):
    """ 
    Find stagnation pressure ratio across three oblique shocks
    Parameters:
        M1: incoming Mach number (unitless)
        gamma: ratio of specific heats (unitless)
        th1: first shock angle in degrees
        th2: second shock angle in degrees
        total_deg: total turning angle in degrees (default 20 degrees)
    Returns:
        P04/P01: stagnation pressure ratio across three oblique shocks (unitless)
    """
    th3 = total_deg - th1 - th2
    if th3 <= 0:
        return np.nan
    P02_P01_val, M2 = P02_P01(M1, gamma, th1)
    P03_P02_val, M3 = P02_P01(M2, gamma, th2)
    P04_P03_val, M4 = P02_P01(M3, gamma, th3)
    return P02_P01_val * P03_P02_val * P04_P03_val

# Grid over first two step angles
N = 300 # number of points per axis
theta1 = np.linspace(TOL, TOTAL_TURN - TOL, N)
theta2 = np.linspace(TOL, TOTAL_TURN - TOL, N)
T1, T2 = np.meshgrid(theta1, theta2, indexing='xy')

# Evaluate surface
vec_P = np.vectorize(lambda a, b: P04_P01_three(M_1, gamma, a, b, TOTAL_TURN))
Z = vec_P(T1, T2)

# Mask invalid areas where theta3 <= 0 (i.e. theta1 + theta2 >= 20º)
Z = np.ma.masked_where(T1 + T2 >= TOTAL_TURN - TOL, Z)

# Plot
plt.figure(figsize=(7,6))
cf = plt.contourf(T1, T2, Z, levels=60)
plt.colorbar(cf, label=r"$P_{0,4}/P_{0,1}$")
# Boundary line theta1 + theta2 = TOTAL_TURN (else its invalid (negative theta3))
x = np.linspace(0, TOTAL_TURN, 200)
plt.plot(x, TOTAL_TURN - x, 'k-', linewidth=1)
plt.xlim(0, TOTAL_TURN)
plt.ylim(0, TOTAL_TURN)
plt.xlabel(r"First step angle $\theta_1$ (deg)")
plt.ylabel(r"Second step angle $\theta_2$ (deg)")
plt.title("Three oblique shocks, total turn = 20°: $P_{0,4}/P_{0,1}$")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 3D surface from the side (not bird's-eye)
Z_filled = np.where(np.ma.getmaskarray(Z), np.nan, Z)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T1, T2, Z_filled, rstride=2, cstride=2,
                       linewidth=0, antialiased=True, cmap='viridis')

# View not top-down
ax.view_init(elev=30, azim=-135)

ax.set_xlabel(r"$\theta_1$ (deg)")
ax.set_ylabel(r"$\theta_2$ (deg)")
ax.set_zlabel(r"$P_{0,4}/P_{0,1}$")
ax.set_title("Three oblique shocks, total turn = 20°")

fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1, label=r"$P_{0,4}/P_{0,1}$")
plt.tight_layout()
plt.show()
