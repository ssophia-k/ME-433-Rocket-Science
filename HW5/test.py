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

gamma = gamma_air
A_star = 0.0001094080858263044
M = inverse_area_mach_relation(2, gamma)  # subsonic

print(M)
