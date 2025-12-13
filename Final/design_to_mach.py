import numpy as np
import pickle
import pandas as pd
from matplotlib import pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.misc_functions import get_speed_of_sound
from Tools.constants import R_air, gamma_air
from Tools.isentropic import *

from inlet import inlet as Inlet
from diffuser import find_diffuser, evaluate_diffuser
from flameholder import flameholder
from combustor import solve_combustor_length, evaluate_combustor
from converging_section import design_converging_section, analyze_converging_section
from nozzle import design_nozzle, analyze_nozzle_moc
from thrust_calc import calculate_thrust
from plot_top import plot_top
from plot_bottom import plot_bottom


# Constants
P_atm = 9112.32
T_atm = 216.65
M_max = 3.25
gamma = gamma_air
Resolution = 100

# Design Parameters
m_dot = 10
width = 1
turn_angles = [10, 10, 10]
M_exit_diffuser = 0.1
diffuser_length = 0.1
m_dot_fuel = 0.1
converging_length = 0.1

# Design
inlet = Inlet(P_atm, T_atm, M_max, m_dot, turn_angles, width=width)
M1, P1, T1, _, _, _ = inlet.output_properties(P_atm, T_atm, M_max)
diffuser_df = find_diffuser(M1, P1, T1, m_dot, width, M_exit_diffuser, diffuser_length, Resolution)
M2 = diffuser_df['Mach'].iloc[-1]
P2 = diffuser_df['Pressure'].iloc[-1]
T2 = diffuser_df['Temperature'].iloc[-1]
P3, M3 = flameholder(P2, M2)
T3 = T2
combustor_dict = solve_combustor_length(M3, P3, T3, m_dot, width, m_dot_fuel)
P4 = combustor_dict["P_out"]
T4 = combustor_dict["T_out"]
M4 = combustor_dict["M_out"]
P5s, T5s, M5s, _, A5s, h5s, x5s = design_converging_section(P4, T4, M4, m_dot, converging_length, width)
P5 = P5s[-1]
T5 = T5s[-1]
M5 = M5s[-1]
converge_df = pd.DataFrame({"height": h5s, "x_vals": x5s})
P6s, T6s, M6s, _, A6s, h6s, x6s, _, _ = design_nozzle(P5, T5, M5, m_dot, P_atm, width)
nozzle_df = pd.DataFrame({"height": h6s, "x_vals": x6s})

# Evaluate across Mach range
M_atms = np.linspace(2.75, 3.25, 51)
thrusts = []

_, _, top_profile_back_thickness = plot_top(inlet, 0, 0, inlet.xs[-1] + diffuser_length + combustor_dict["length_m"] + converging_length + x6s[-1])
_, _, length_of_front, angle_of_front = plot_bottom(inlet, diffuser_df, combustor_dict, x5s, h5s, x6s, h6s)

results = []

for M_in in M_atms:
    P_inlet, T_inlet, _, M_inlet = inlet.get_1d_profiles(inlet.xs, P_atm, T_atm, M_in)
    M_normal, P_normal, T_normal, _, _, _ = inlet.output_properties(P_atm, T_atm, M_in)
    diffuser_results = evaluate_diffuser(diffuser_df['x'], diffuser_df['y'], M_normal, P_normal, T_normal, width)
    P_out, _ = flameholder(diffuser_results['Pressure'].iloc[-1], diffuser_results['Mach'].iloc[-1], gamma)
    u_in = diffuser_results['Mach'].iloc[-1] * get_speed_of_sound(diffuser_results['Temperature'].iloc[-1])
    rho_in = P_out / (R_air * diffuser_results['Temperature'].iloc[-1])
    area_in = diffuser_df['y'].iloc[-1] * width
    m_dot_air = area_in * rho_in * u_in
    combustor_results = evaluate_combustor(combustor_dict["length_m"], diffuser_results['Mach'].iloc[-1], P_out, diffuser_results['Temperature'].iloc[-1], m_dot_air, width, combustor_dict["M_out"])
    con_Ps, con_Ts, con_Ms = analyze_converging_section(converge_df["height"], combustor_results["P_out"], combustor_results["T_out"], combustor_results["M_out"], width)
    noz_Ps, noz_Ts, noz_Ms, _, _ = analyze_nozzle_moc(nozzle_df["x_vals"], nozzle_df["height"], con_Ps[-1], con_Ts[-1], con_Ms[-1], width, n_characteristics=50)
    
    thrust = calculate_thrust(inlet, P_atm, M_in, T_atm, noz_Ps[-1], noz_Ms[-1], noz_Ts[-1], nozzle_df["height"].iloc[-1] * width, length_of_front, angle_of_front, top_profile_back_thickness, width)
    thrusts.append(thrust)

    results.append({
        "M_in": M_in,
        "Thrust": thrust
    })

avg_thrust = sum(thrusts) / len(thrusts)
print(f"Average Thrust: {avg_thrust} N")