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

# # Atmosphere:
# P_atm = 9112.32  # Pa
# T_atm = 216.65  # K
# M_atms = np.linspace(2.75, 3.25, 5)
# gamma = gamma_air

# # Basic properties:
# width = 1  # m

def evaluate_up_to_combustor(M_lowest, inlet_design_params_dict, diffuser_df):
    # Atmosphere:
    P_atm = 9112.32  # Pa
    T_atm = 216.65  # K
    M_atms = [M_lowest]
    gamma = gamma_air
    width = 1

    # with open("Final/profiles/inlet_design_params_dict.pkl", "rb") as f:
    #     inlet_design_params_dict = pickle.load(f)
    # diffuser_df = pd.read_pickle("Final/profiles/diffuser_df.pkl")
    # with open("Final/profiles/combustor_dict.pkl", "rb") as f:
    #     combustor_dict = pickle.load(f)
    # converge_df = pd.read_pickle("Final/profiles/converge_df.pkl")
    # nozzle_df = pd.read_pickle("Final/profiles/nozzle_df.pkl")

    # Inlet Parameters
    width = inlet_design_params_dict["width"] # m
    inlet = Inlet(**inlet_design_params_dict)
    inlet_xs = inlet.xs
    inlet_ys = inlet.ys

    results = []
    for M_in in M_atms:
        results.append({
            "M_in": M_in,
            "xs": [],
            "Ms": [],
            "Ps": [],
            "Ts": [],
            "P0s": [],
            "T0s": [],
            "Ss": [],
            "Thrust": None,
            "fuel_info": {}
        })

    for r in results:
        M_in = r["M_in"]
        print(f"Running {M_in}")

        # Inlet
        P_inlet, T_inlet, _, M_inlet, P0_inlet, T0_inlet, s_inlet = inlet.get_1d_profiles(inlet_xs, P_atm, T_atm, M_in)
        r["xs"].extend(inlet_xs)
        r["Ms"].extend(M_inlet)
        r["Ps"].extend(P_inlet)
        r["Ts"].extend(T_inlet)
        r["P0s"].extend(P0_inlet)
        r["T0s"].extend(T0_inlet)
        r["Ss"].extend(s_inlet)
        # r["P0s"].extend(P_inlet * P0_P(M_inlet, gamma))
        # r["T0s"].extend(T_inlet * T0_T(M_inlet, gamma))
        M_normal, P_normal, T_normal, _, _, _ = inlet.output_properties(P_atm, T_atm, M_in)
        r["xs"].append(r["xs"][-1])
        r["Ms"].append(M_normal)
        r["Ps"].append(P_normal)
        r["Ts"].append(T_normal)
        r["P0s"].append(P_normal * P0_P(M_normal, gamma))
        r["T0s"].append(T_normal * T0_T(M_normal, gamma))

        # Diffuser
        diffuser_results = evaluate_diffuser(diffuser_df['x'], diffuser_df['y'], r["Ms"][-1], r["Ps"][-1], r["Ts"][-1], width)
        r["xs"].extend(diffuser_results['x'] + r["xs"][-1])
        r["Ms"].extend(diffuser_results['Mach'])
        r["Ps"].extend(diffuser_results['Pressure'])
        r["Ts"].extend(diffuser_results['Temperature'])
        r["P0s"].extend(diffuser_results['Stag Pressure'])
        r["T0s"].extend(diffuser_results['Stag Temperature'])

        # Flameholder
        P_out, _ = flameholder(r["Ps"][-1], r["Ms"][-1], gamma)
        r["xs"].append(r["xs"][-1])
        r["Ms"].append(r["Ms"][-1])
        r["Ps"].append(P_out)
        r["Ts"].append(r["Ts"][-1])
        r["P0s"].append(P_out * P0_P(r["Ms"][-1], gamma))
        r["T0s"].append(r["T0s"][-1])
    
    return results[0]["Ms"][-1], results[0]["Ps"][-1], results[0]["Ts"][-1]