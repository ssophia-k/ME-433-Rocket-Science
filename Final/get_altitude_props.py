import numpy as np

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.constants import *
from Tools.standard_atmosphere import *

height_ft = 55000 # ft
height_km = 0.0003048 * height_ft # km

T = get_temp_from_altitude(height_km)
P = get_pressure_from_altitude(height_km)

print(f"Temperature: {T} K")
print(f"Pressure: {P} Pa")

# Temperature: 216.64999999999998 K
# Pressure: 9112.320878910645 Pa