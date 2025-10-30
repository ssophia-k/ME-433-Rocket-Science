import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import seaborn as sns

from Tools.constants import *
from Midterm.many_shapes import *
from Midterm.cost import *
from Midterm.Simulate.viscous_shapes import *

# Givens
M = 3
gamma = gamma_air
P = P_sea  # Pa
T = 300  # K
L = 1  # m, depth into page
lowest_per_shape = {}

parabola(M, gamma, P, T, L, False, lowest_per_shape)
triangle(M, gamma, P, T, L, False, lowest_per_shape)
triangular_wedge(M, gamma, P, T, L, False, lowest_per_shape)
trapezoid(M, gamma, P, T, L, False, lowest_per_shape)
power_series(M, gamma, P, T, L, False, lowest_per_shape)
diamond(M, gamma, P, T, L, False, lowest_per_shape)

print(lowest_per_shape)

lowest_shape = min(lowest_per_shape.items(), key=lambda x: x[1]['cost_star'])
name, params = lowest_shape

print(f"Lowest cost shape: {name}")
print(f"Cost: {params['cost_star']}")

"""
Lowest cost shape: triangular_wedge
Cost: 14674613.097952837
"""