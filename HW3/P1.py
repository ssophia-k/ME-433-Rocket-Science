"""
1. Write a Matlab function which takes in the incoming Mach number, ratio of specific
heats, and turning angle, and outputs the corresponding pressure ratio, temperature ratio,
and outgoing Mach number solution for a calorically perfect gas.
"""

import numpy as np

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.constants import *
from Tools.oblique_shock import *

gamma = gamma_air
M_in = 5
theta_deg = 30

# this is some testing to make sure we match up with online calculator
beta, P2_1, T2_1, M2 = mach_function(M_in, gamma, theta_deg)
print(f"For M_in={M_in}, theta={theta_deg} deg, gamma={gamma}:")
print(f"Shock angle (beta) = {beta:.3f} deg")
print(f"Pressure ratio (P2/P1) = {P2_1:.3f}")
print(f"Temperature ratio (T2/T1) = {T2_1:.3f}")
print(f"Outgoing Mach number (M2) = {M2:.3f}")
