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
from Midterm.cost import cost_function

# Givens
M_in = 3
gamma = 1.4
aoas = np.linspace(-5, 5, 1001)  # angle of attack from -5 to 5 degrees

# airfoil geometry, all symmetric(rows: xposition, yposition) 
x1, y1 = np.loadtxt("HW4/airfoils/airfoil1.txt", delimiter=",")
x2, y2 = np.loadtxt("HW4/airfoils/airfoil2.txt", delimiter=",")
x3, y3 = np.loadtxt("HW4/airfoils/airfoil3.txt", delimiter=",")
x4, y4 = np.loadtxt("HW4/airfoils/airfoil4.txt", delimiter=",")

airfoils = {
    "airfoil1": (x1, y1),
    "airfoil2": (x2, y2),
    "airfoil3": (x3, y3),
    "airfoil4": (x4, y4),
}

thetas_top = get_turn_angles(x4, y4, 5, top=True)
thetas_bot = get_turn_angles(x4, -y4, 5, top=False)

#M_top, P_top, M_bot, P_bot = get_Mach_numbers_p_ratios(M_in, gamma, thetas_top, thetas_bot)



# plt.figure()
# plt.plot(x4, M_top)
# # plt.plot(x1[1:], P_top)
# plt.plot(x4, M_bot)
# # plt.plot(x1[1:], P_bot)
# plt.show()


plt.figure()
for name, (x, y) in airfoils.items():
    cds, cls = [], []
    cost = []
    for a in aoas:
        cl, cd = get_cL_and_cD(M_in, gamma, x, y, a)
        cls.append(cl); cds.append(cd)
        # cost.append(cost_function(x, y, -y, M_in, gamma, P_inf=101325, T_inf=288.15, depth=1.0, inviscid=True)[2])
    plt.plot(cds, cls, label=name, marker='o', ms=0.5, lw=1)
    # plt.plot(aoas, cost, label=f"cost_{name}", linestyle='--')

plt.xlabel(r"$c_D$")
plt.ylabel(r"$c_L$")
plt.title("Lift coefficient vs Drag coefficient, M=3, γ=1.4")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



