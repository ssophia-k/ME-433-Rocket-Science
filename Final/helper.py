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
from diffuser import evaluate_diffuser
from flameholder import flameholder
from combustor import evaluate_combustor
from converging_section import analyze_converging_section
from nozzle import analyze_nozzle

# Atmosphere:
P_atm = 9112.32  # Pa
T_atm = 216.65  # K
M_atms = np.linspace(2.75, 3.25, 5)
gamma = gamma_air

# Basic properties:
width = 1  # m

def evaluate_up_to_converger(M_lowest, inlet_design_params_dict, diffuser_df, combustor_dict):
    inlet = Inlet(**inlet_design_params_dict)
    inlet_xs = inlet.xs
    inlet_ys = inlet.ys
    r = {
        "xs": [],
        "Ms": [],
        "Ps": [],
        "Ts": [],
    }
    M_in = M_lowest
    # Inlet
    P_inlet, T_inlet, _, M_inlet = inlet.get_1d_profiles(inlet_xs, P_atm, T_atm, M_in)
    P0_inlet = P_inlet * P0_P(M_inlet, gamma)
    T0_inlet = T_inlet * T0_T(M_inlet, gamma)
    r["xs"].extend(inlet_xs)
    r["Ms"].extend(M_inlet)
    r["Ps"].extend(P_inlet)
    r["Ts"].extend(T_inlet)
    M_normal, P_normal, T_normal, _, _, _ = inlet.output_properties(P_atm, T_atm, M_in)
    r["xs"].append(r["xs"][-1])
    r["Ms"].append(M_normal)
    r["Ps"].append(P_normal)
    r["Ts"].append(T_normal)
    # Diffuser
    diffuser_results = evaluate_diffuser(diffuser_df['x'], diffuser_df['y'], r["Ms"][-1], r["Ps"][-1], r["Ts"][-1], width)
    r["xs"].extend(diffuser_results['x'] + r["xs"][-1])
    r["Ms"].extend(diffuser_results['Mach'])
    r["Ps"].extend(diffuser_results['Pressure'])
    r["Ts"].extend(diffuser_results['Temperature'])
    # Flameholder
    P_out, _ = flameholder(r["Ps"][-1], r["Ms"][-1], gamma)
    r["xs"].append(r["xs"][-1])
    r["Ms"].append(r["Ms"][-1])
    r["Ps"].append(P_out)
    r["Ts"].append(r["Ts"][-1])
    # Combustor
    u_in =  r["Ms"][-1]* get_speed_of_sound(r["Ts"][-1])
    rho_in = r["Ps"][-1] / (R_air * r["Ts"][-1])
    area_in = diffuser_df['y'].iloc[-1] * width
    m_dot_air = area_in * rho_in * u_in
    combustor_results = evaluate_combustor(combustor_dict["length_m"], r["Ms"][-1], r["Ps"][-1], r["Ts"][-1], m_dot_air, width, combustor_dict["m_dot_fuel"])
    r["xs"].append(combustor_results["length_m"] + r["xs"][-1])
    r["Ms"].append(combustor_results["M_out"])
    r["Ps"].append(combustor_results["P_out"])
    r["Ts"].append(combustor_results["T_out"])
    
    return r["Ms"][-1], r["Ps"][-1], r["Ts"][-1]