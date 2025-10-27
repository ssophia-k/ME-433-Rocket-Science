"""
1. Plot as a function of x the pressure distribution normalized by the stagnation pressure for
the attached nozzle profile. The profile is given in terms of a two column matrix where
the first column is the x location and the second column is the radius of an axisymmetric
circular nozzle through which the flow goes through. Assume air and CPG.
a. Case 1: fully subsonic, you may choose a back pressure
b. Case 2: fully subsonic, with the throat at sonic, what is the back pressure?
c. Case 3: Normal shock at x = 0.75 in the nozzle, what is the back pressure?
d. Case 4: Normal shock at the exit of the nozzle, what is the back pressure?
e. Case 5: Ideally expanded, what is the back pressure?
f. What pressure range corresponds to oblique shockwaves at the exit of the nozzle?
g. What pressure range corresponds to expansion waves at the exit of the nozzle?
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.constants import *
from HW5.helpers import *
from Tools.isentropic import *
from Tools.normal_shock import *

gamma = gamma_air

x_shock = 0.75

# Load data from the file
data = np.loadtxt('HW5/PNozzle.txt', delimiter=',')
df = pd.DataFrame(data, columns=['x_location', 'radius']) # Convert to DataFrame (easier imo)
df['area'] = np.pi * df['radius']**2
#print(df.head()) # test

# plt.plot(df['x_location'],df['area'])
# plt.show()

# Know throat is sonic. M=1, so A_min = A*
A_star = df['area'].min()
x_star = df.loc[df['area'] == A_star, 'x_location'].iloc[0]

# Now find M at every other point
Mach_list = []
P_P0_list = []
P01_P02 = 0 # used to normalize later after N.S

A_star_new = 0 # A* post shock

# Iterate till we get to area = A_star. that will be subsonic sln.
# Then A_star to x = 0.75 will be supersonic sln.
# we also know that x = 0.75 is explicitly in the df. so helpful.
# Stop at 0.75 = x. need to apply normal shock
for x, area in zip(df['x_location'], df['area']):
    
    if x <= x_star:
        A_ratio_sq = (area / A_star)**2
        # subsonic up to throat
        M = inverse_area_mach_relation(A_ratio_sq, gamma)[0]
        P_P0 = 1.0 / P0_P(M, gamma)

    elif x < x_shock:
        A_ratio_sq = (area / A_star)**2
        # supersonic to x=0.75
        M = inverse_area_mach_relation(A_ratio_sq, gamma)[1]
        P_P0 = 1.0 / P0_P(M, gamma)

    elif x == x_shock:
        A_ratio_sq = (area / A_star)**2
        # normal shock exactly at x=0.75
        shock_row = df.loc[df['x_location'] == 0.75]
        A_ratio_sq_pre = (area / A_star) ** 2 
        M1 = inverse_area_mach_relation(A_ratio_sq_pre, gamma)[1]
        M2 = M2_from_normal_shock(M1, gamma)

        # find new stag pressure to normalize by
        P01_P1 = P0_P(M1, gamma)
        P1_P2 = 1.0/ P2_P1_from_normal_shock(M1, gamma)
        P2_P02 = 1.0/ P0_P(M2, gamma)
        P01_P02 = P01_P1 * P1_P2 * P2_P02

        # find new A* as well
        A_A_star = np.sqrt(area_mach_relation(M2, gamma))
        A_star_new = area/A_A_star

        M = M2
        P_P0 = (1.0 / P0_P(M2, gamma))/ P01_P02

    elif x>x_shock:
        A_ratio_sq = (area / A_star_new)**2
        # subsonic after shock bc we expand back up
        M = inverse_area_mach_relation(A_ratio_sq, gamma)[0]
        P_P0 = (1.0 / P0_P(M, gamma)) / P01_P02 # renormalize w new stag pressure after N.S.

    Mach_list.append(M)
    P_P0_list.append(P_P0)


# Add results to DataFrame
df['Mach'] = Mach_list
df['P_P0'] = P_P0_list

df.to_csv('HW5/1c_debug.txt', sep='\t', index=False, float_format='%.6f')


print(f"Final P/P0: {P_P0_list[-1]}")
# Final P/P0: 0.2133548440278774

plt.plot(df['x_location'], df['P_P0'])
# plt.plot(df['x_location'], df['area'], label = "area")
# plt.plot(df['x_location'], df['Mach'], label = "Mach")

plt.xlabel('x location')
plt.ylabel('P/P0')
plt.title('Normalized Pressure Distribution along Nozzle Given N.S. at x=0.75')
plt.grid(True)
plt.show()
