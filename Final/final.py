import numpy as np
import pickle
import pandas as pd
from matplotlib import pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))


from inlet import inlet as Inlet
from diffuser import find_diffuser
from flameholder import flameholder
from combustor import solve_combustor_length
from converging_section import design_converging_section
from nozzle import design_nozzle
from thrust_calc import calculate_thrust
from plot_top import plot_top
from plot_bottom import plot_bottom


# Atmosphere:
P_atm = 9112.32  # Pa
T_atm = 216.65  # K
M_max = 3.25
M_atm = M_max
M_lowest = 2.75


# Basic properties:
m_dot = 10  # kg/s
width = 1  # m

# Inlet:
turn_angles = [10, 10, 10]  # turn angles of inlet, degrees
inlet = Inlet(P_atm, T_atm, M_max, m_dot, turn_angles, width=width)
inlet_design_params_dict = {
    'P_atm': P_atm,
    'T_atm': T_atm,
    'M_max': M_max,
    'm_dot': m_dot,
    'turn_angles': turn_angles,
    'width': width
}
with open('Final/profiles/inlet_design_params_dict.pkl', 'wb') as f:
    pickle.dump(inlet_design_params_dict, f)

# Diffuser:
M_exit_diffuser = 0.1  #
diffuser_length = 0.1  # length of diffuser, m
Resolution = 100

# combustor:
m_dot_fuel = 0.1  # kg/s

# Converging section:
converging_length = 0.1  # m

# Nozzle:
P_exit_nozz = P_atm

M1, P1, T1, _, _, _ = inlet.output_properties(P_atm, T_atm, M_atm)

diffuser_df = find_diffuser(M1, P1, T1, m_dot, width, M_exit_diffuser, diffuser_length, Resolution)
M2 = diffuser_df['Mach'].iloc[-1]
P2 = diffuser_df['Pressure'].iloc[-1]
T2 = diffuser_df['Temperature'].iloc[-1]
diffuser_df.to_pickle('Final/profiles/diffuser_df.pkl')

P3, M3 = flameholder(P2, M2)
T3 = T2

# might need to design combustor for lowest mach to avoid choke. or we can just guess-n-check.

combustor_dict = solve_combustor_length(M3, P3, T3, m_dot, width, m_dot_fuel)
if combustor_dict["is_choked"]:
    print(f"Choked at {M_atm=}, {m_dot_fuel=}")
P4 = combustor_dict["P_out"]
T4 = combustor_dict["T_out"]
M4 = combustor_dict["M_out"]
with open('Final/profiles/combustor_dict.pkl', 'wb') as f:
    pickle.dump(combustor_dict, f)

# # need to design converging section for lowest mach operation s.t. we always get to M=1 before nozzle
# M4_lowest, P4_lowest, T4_lowest = evaluate_up_to_converger(M_lowest, inlet_design_params_dict, diffuser_df, combustor_dict)

P5s, T5s, M5s, _, A5s, h5s, x5s = design_converging_section(P4, T4, M4, m_dot, converging_length, width)
P5 = P5s[-1]
T5 = T5s[-1]
M5 = M5s[-1]
converge_df = pd.DataFrame({
    "Pressure": P5s,
    "Temperature": T5s,
    "Mach": M5s,
    "Area": A5s,
    "height": h5s,
    "x_vals": x5s,
})
converge_df.to_pickle('Final/profiles/converge_df.pkl')

P6s, T6s, M6s, _, A6s, h6s, x6s, _, _ = design_nozzle(P5, T5, M5, m_dot, P_exit_nozz, width)
P6 = P6s[-1]
T6 = T6s[-1]
M6 = M6s[-1]
nozzle_df = pd.DataFrame({
    "Pressure": P6s,
    "Temperature": T6s,
    "Mach": M6s,
    "Area": A6s,
    "height": h6s,
    "x_vals": x6s,
})
nozzle_df.to_pickle('Final/profiles/nozzle_df.pkl')

# Plot Profile
print(f"Inlet height: {(inlet.y_lip-0)}")
print(f"Outlet height: {h6s[-1]}")
inlet_length = inlet.xs[-1]
combustor_length = combustor_dict["length_m"]
nozzle_length = x6s[-1]-x6s[0]
total_length = inlet_length+diffuser_length+combustor_length+converging_length+nozzle_length

fig, ax = plt.subplots(figsize=(12, 6))

# Plot the top surface
inner_coords, outer_coords, top_profile_back_thickness = plot_top(
    ax, 
    inlet, 
    x_offset=0,
    y_offset=0,
    x_end=total_length,
)

# Plot the bottom surface
top_face, bottom_face, length_of_front, angle_of_front = plot_bottom(
    ax,
    inlet,
    diffuser_df,
    combustor_dict,
    x5s,
    h5s,
    x6s,
    h6s
)

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Complete Ramjet Geometry')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

plt.tight_layout()
plt.show()

# Calculate Thrust
thrust = calculate_thrust(inlet, P_atm, M_atm, T_atm, P6, M6, T6, A6s[-1], length_of_front, angle_of_front, top_profile_back_thickness, width)
print(f"{thrust=} N")

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
