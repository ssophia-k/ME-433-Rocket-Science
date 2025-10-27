import numpy as np
import matplotlib.pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.constants import *
from Tools.standard_atmosphere import *
from Tools.misc_functions import *
from Tools.numerical_iterator import *

# So we know that we need T/P > 0.63956 K/Pa in order to get Kn>0.1 (free molecular flow)
# IDK how this function acts so we plot it

GOAL = 0.63956 # K/Pa

altitude = np.linspace(0, 47, 1000) # km
pressure = np.array([get_pressure_from_altitude(h) for h in altitude]) # Pa
temperature = np.array([get_temp_from_altitude(h) for h in altitude]) # K
T_over_P = temperature / pressure # K/Pa

plt.plot(altitude, T_over_P, label="T/P [K/Pa]", color='b')
plt.axhline(y=(GOAL), color='r', linestyle='--', label=f"Goal: {GOAL} K/Pa")
plt.xlabel("Altitude [km]")
plt.ylabel("T/P [K/Pa]")
plt.title("T/P vs Altitude")
plt.grid(True)
plt.legend()
plt.show()

# from inspection, this only crosses once. which is good for our numerical iterator YAY

def T_P_ratio(altitude):
    return get_temp_from_altitude(altitude) / get_pressure_from_altitude(altitude)


print(numerical_iterator(func=T_P_ratio, start=0, end=47, goal_y=GOAL))
# Answer is 37.66 km, which matches the plot


