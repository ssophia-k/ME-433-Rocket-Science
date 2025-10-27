import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.constants import *
from Tools.standard_atmosphere import *
from Tools.misc_functions import *
from Tools.numerical_iterator import *

R_air = R_air # J/(kg*K)
Volume = 8181.23 # m^3 (from the homework pdf)
m = 600 # kg (from the homework pdf)

def function(height, balloon_temp):
    P_atm = get_pressure_from_altitude(height) # Pa
    T_atm = get_temp_from_altitude(height) # K
    return (Volume * P_atm) / (R_air * T_atm) - (Volume * P_atm) / (R_air * balloon_temp) - m # needs to equal 0

altitude = np.linspace(0, 47, 1000) # km

# Choose to iterate balloon temps from T_sea to 10000 K (in order to see end behavior)
balloon_temps = np.linspace(T_sea, 10000, 1000) # K
valid_h = np.empty_like(balloon_temps, dtype=float)
valid_h[:] = np.nan # Initialize with NaNs since IDK which temps will return a valid height and which won't

for i, T in enumerate(balloon_temps):
    try:
        valid_h[i] = numerical_iterator(
            func=lambda h, TT=T: function(h, TT),
            start=0.0, end=47.0, goal_y=0.0
        )
    except ValueError:
        # No solution for this T; leave as NaN (won't be plotted)
        pass

plt.figure()
plt.plot(balloon_temps, valid_h)
plt.xlabel("Balloon temperature [K]")
plt.ylabel("Equilibrium altitude [km]")
plt.title("Equilibrium Height vs. Balloon Temperature (solvable cases only)")
plt.grid(True)
plt.show()

# min balloon height and temp for equilibrium
min_temp = balloon_temps[np.nanargmin(valid_h)]
min_height = np.nanmin(valid_h)
print(f"Minimum balloon temperature for equilibrium: {min_temp:.2f} K at height {min_height:.2f} km")

# max balloon height and temp for equilibrium
max_temp = balloon_temps[np.nanargmax(valid_h)]
max_height = np.nanmax(valid_h)
print(f"Maximum balloon temperature for equilibrium: {max_temp:.2f} K at height {max_height:.2f} km")


# From the visual plot, it looks exponential, so try to fit it with that with scipy (learned this from another class lol)
# This means the form is: h = h_inf - A * exp(-T/tau)
# where h_inf is the asymptotic height, A is a scaling factor, and tau is the "time constant" (Temp constant in this case?)
mask = ~np.isnan(valid_h) # only fit the valid points (no NANs)
T_fit = balloon_temps[mask]
h_fit = valid_h[mask]

def sat_func(T, h_inf, A, tau): # this is the function form we are fitting to
    return h_inf - A * np.exp(-T / tau)

popt, pcov = curve_fit(sat_func, T_fit, h_fit, p0=[21, 100, 300])  
# p0 is initial guess. h_inf looks like 21km from the plot. A = 100 was another guess. tau = 300 was another guess.
# we see later that A and tau change quite a bit. which is fine since they are just seed points for scipy to look from.
h_inf, A, tau = popt
print("h_inf =", h_inf, "A =", A, "tau =", tau) # Fitted parameters

T_dense = np.linspace(T_fit.min(), T_fit.max(), 1000)
plt.title("Equilibrium Height vs. Balloon Temperature Plus Exponential Fit")
plt.plot(T_fit, h_fit, '.', alpha=0.4, label="data")
plt.plot(T_dense, sat_func(T_dense, *popt), lw=2, label="exponential fit")
plt.xlabel("Balloon temperature [K]")
plt.ylabel("Equilibrium altitude [km]")
plt.legend(); plt.grid(True); plt.show()
#From this plot it isn't the BEST but I think it is okay?

