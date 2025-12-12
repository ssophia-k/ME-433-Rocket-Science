import numpy as np

from matplotlib import pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.misc_functions import get_speed_of_sound

from inlet import inlet as Inlet
from diffuser import find_diffuser
from flameholder import flameholder
from combuster import solve_combustor_length
from converging_section import converging_section
from nozzle import nozzle

from plot_top import plot_top

# Atmosphere:
P_atm = 9112.32  # Pa
T_atm = 216.65  # K
M_atms = np.linspace(2.75, 3.25)
M_atms = [3.25]

# Basic properties:
m_dot = 10  # kg/s
width = 1  # m

# Inlet:
M_max = M_atms[-1]
turn_angles = [10, 10, 10]  # turn angles of inlet, degrees
inlet = Inlet(P_atm, T_atm, M_max, m_dot, turn_angles, width=width)

# Diffuser:
M_exit_diffuser = 0.1  #
diffuser_length = 0.05  # length of diffuser, m
Resolution = 100

# Combustor:
m_dot_fuel = 0.1  # kg/s

# Converging section:
converging_length = 0.1  # m

# Nozzle:
P_exit_nozz = P_atm

for M_atm in M_atms:
    M1, P1, T1, _, _, _ = inlet.output_properties(P_atm, T_atm, M_atm)
    
    diffuser_df = find_diffuser(M1, P1, T1, m_dot, width, M_exit_diffuser, diffuser_length, Resolution)
    M2 = diffuser_df['Mach'].iloc[-1]
    P2 = diffuser_df['Pressure'].iloc[-1]
    T2 = diffuser_df['Temperature'].iloc[-1]
    
    P3, M3 = flameholder(P2, M2)
    T3 = T2
    
    combustor_dict = solve_combustor_length(M3, P3, T3, m_dot, width, m_dot_fuel)
    if combustor_dict["is_choked"]:
        print(f"Choked at {M_atm=}, {m_dot_fuel=}")
    P4 = combustor_dict["P_out"]
    T4 = combustor_dict["T_out"]
    M4 = combustor_dict["M_out"]
    
    P5s, T5s, M5s, _, A5s, h5s, x5s = converging_section(P4, T4, M4, m_dot, converging_length, width)
    P5 = P5s[-1]
    T5 = T5s[-1]
    M5 = M5s[-1]
    
    P6s, T6s, M6s, _, A6s, h6s, x6s = nozzle(P5, T5, M5, m_dot, P_exit_nozz, width)
    P6 = P6s[-1]
    T6 = T6s[-1]
    M6 = M6s[-1]

thrust_estimate = (P6*A6s[-1] + m_dot*get_speed_of_sound(T6)*M6) - (P_atm*(inlet.y_lip-0)*width+m_dot*get_speed_of_sound(T_atm)*M_atm)
print(f"{thrust_estimate=} N")


print(f"{M1=}")
print(f"{T1=}")
print(f"{P1=}")
print(f"{M2=}")
print(f"{T2=}")
print(f"{P2=}")
print(f"{M3=}")
print(f"{T3=}")
print(f"{P3=}")
print(f"{M4=}")
print(f"{T4=}")
print(f"{P4=}")
print(f"{M5=}")
print(f"{T5=}")
print(f"{P5=}")
print(f"{M6=}")
print(f"{T6=}")
print(f"{P6=}")

print(f"Inlet height: {(inlet.y_lip-0)}")
print(f"Outlet height: {h6s[-1]}")

inlet_length = inlet.xs[-1]
combustor_length = combustor_dict["length_m"]
nozzle_length = x6s[-1]-x6s[0]

total_length = inlet_length+diffuser_length+combustor_length+converging_length+nozzle_length
print(f"Total length: {total_length} m")