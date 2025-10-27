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

# Load data from the file
data = np.loadtxt('HW5/PNozzle.txt', delimiter=',')
df = pd.DataFrame(data, columns=['x_location', 'radius']) # Convert to DataFrame (easier imo)
df['area'] = np.pi * df['radius']**2
#print(df.head()) # test

# Know throat is sonic. M=1, so A_min = A*
A_star = df['area'].min()

# Now find M at every other point given that we know A_star
Mach_list = []
P_P0_list = []

for area in df['area']:
    A_ratio_sq = (area / A_star)**2
    try:
        M = inverse_area_mach_relation(A_ratio_sq, gamma)[0]  # subsonic
    except (RuntimeError, ValueError):
        M = float('nan') 
    Mach_list.append(M)
    P_P0_list.append(1 / P0_P(M, gamma))

# Add results to DataFrame
df['Mach'] = Mach_list
df['P_P0'] = P_P0_list

print(f"Final P/P0: {P_P0_list[-1]}")
# Final P/P0: 0.9982308815091709

plt.plot(df['x_location'], df['P_P0'])
plt.xlabel('x location')
plt.ylabel('P/P0')
plt.title('Normalized Pressure Distribution along Nozzle Given M=1 at Throat')
plt.grid(True)
plt.show()

# Back pressure is ~= P0 since Pe/P0 goes to 1 on the graph
