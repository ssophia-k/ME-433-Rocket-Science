"""
3. Plot on the same graph the lift coefficient as a function of the drag coefficient for the range of
angle of attack from - 5 to 5 degrees. Assume a Mach number of 3 and gamma 1.4.
"""

import numpy as np
import matplotlib.pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from HW4.P3_helpers import *

# Givens
M_in = 3
gamma = 1.4
aoas = np.linspace(-5, 5, 1001)  # angle of attack from -5 to 5 degrees

# airfoil geometry, all symmetric(rows: xposition, yposition) 
x1, y1 = np.loadtxt("HW4/airfoil1.txt", delimiter=",")
x2, y2 = np.loadtxt("HW4/airfoil2.txt", delimiter=",")
x3, y3 = np.loadtxt("HW4/airfoil3.txt", delimiter=",")
x4, y4 = np.loadtxt("HW4/airfoil4.txt", delimiter=",")

airfoils = {
    "airfoil1": (x1, y1),
    "airfoil2": (x2, y2),
    "airfoil3": (x3, y3),
    "airfoil4": (x4, y4),
}

M_top, P_ratios_top, M_bot, P_ratios_bot = get_Mach_numbers_p_ratios(M_in, gamma, x4, y4)

plt.figure()
plt.plot(x4, y4)
#plt.plot(x4[1:], P_ratios_top)
plt.plot(x4[1:], P_ratios_bot)
plt.show()


plt.figure()
for name, (x, y) in airfoils.items():
    cds, cls = [], []
    for a in aoas:
        cl, cd = get_cL_and_cD(M_in, gamma, x, y, a)
        cls.append(cl); cds.append(cd)
    plt.plot(cds, cls, label=name, marker='o', ms=3, lw=1)

plt.xlabel(r"$c_D$")
plt.ylabel(r"$c_L$")
plt.title("Lift coefficient vs Drag coefficient, M=3, γ=1.4")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


