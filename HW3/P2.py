"""
Using this function empirically determine the maximum turning angle for the case of M =
3 and 𝛾 = 1.3
"""

import numpy as np

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.constants import *
from Tools.oblique_shock import *

gamma = 1.3
M_in = 3

max_turning_angle = get_theta_max_beta_from_tbm(M_in, gamma)[0]
print(f"Maximum turning angle for M_in={M_in}, gamma={gamma} is {max_turning_angle:.3f} degrees")

