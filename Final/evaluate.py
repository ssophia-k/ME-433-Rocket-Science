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
M_atms = np.linspace(2.75, 3.25, 3)
gamma = gamma_air

# Basic properties:
width = 1  # m

with open("Final/profiles/inlet_design_params_dict.pkl", "rb") as f:
    inlet_design_params_dict = pickle.load(f)
diffuser_df = pd.read_pickle("Final/profiles/diffuser_df.pkl")
with open("Final/profiles/combustor_dict.pkl", "rb") as f:
    combustor_dict = pickle.load(f)
converge_df = pd.read_pickle("Final/profiles/converge_df.pkl")
nozzle_df = pd.read_pickle("Final/profiles/nozzle_df.pkl")

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
        "fuel_info": {}
    })

for r in results:
    M_in = r["M_in"]

    # Inlet
    P_inlet, T_inlet, _, M_inlet = inlet.get_1d_profiles(inlet_xs, P_atm, T_atm, M_in)
    P0_inlet = P_inlet * P0_P(M_inlet, gamma)
    T0_inlet = T_inlet * T0_T(M_inlet, gamma)
    r["xs"].extend(inlet_xs)
    r["Ms"].extend(M_inlet)
    r["Ps"].extend(P_inlet)
    r["Ts"].extend(T_inlet)
    r["P0s"].extend(P_inlet * P0_P(M_inlet, gamma))
    r["T0s"].extend(T_inlet * T0_T(M_inlet, gamma))
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
    r["P0s"].append(r["Ps"][-1] * P0_P(r["Ms"][-1], gamma))
    r["T0s"].append(r["Ts"][-1] * T0_T(r["Ms"][-1], gamma))
    r["fuel_info"] = {"m_dot_fuel_cmd_kg_s": combustor_results["m_dot_fuel_cmd_kg_s"],
                        "m_dot_fuel_max_kg_s": combustor_results["m_dot_fuel_max_kg_s"],
                        "fuel_limited_by_choke": combustor_results["fuel_limited_by_choke"],
                    }
    
    # Converging section
    con_Ps, con_Ts, con_Ms = analyze_converging_section(converge_df["height"], r["Ps"][-1], r["Ts"][-1], r["Ms"][-1], width)
    r["xs"].extend(converge_df["x_vals"] + r["xs"][-1])
    r["Ms"].extend(con_Ms)
    r["Ps"].extend(con_Ps)
    r["Ts"].extend(con_Ts)
    r["P0s"].extend(con_Ps * P0_P(con_Ms, gamma))
    r["T0s"].extend(con_Ts * T0_T(con_Ms, gamma))

    # Nozzle
    noz_Ps, noz_Ts, noz_Ms = analyze_nozzle(nozzle_df["height"], r["Ps"][-1], r["Ts"][-1], r["Ms"][-1], width)
    r["xs"].extend(nozzle_df["x_vals"] + r["xs"][-1])
    r["Ms"].extend(noz_Ms)
    r["Ps"].extend(noz_Ps)
    r["Ts"].extend(noz_Ts)
    r["P0s"].extend(noz_Ps * P0_P(noz_Ms, gamma))
    r["T0s"].extend(noz_Ts * T0_T(noz_Ms, gamma))


#print(results)

plt.figure()
for r in results:
    plt.plot(r["xs"], r["Ms"], label=f"M_in = {r['M_in']:.2f}")
plt.xlabel("x")
plt.ylabel("Mach number")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('Final/results/mach_profiles.png')

plt.figure()
for r in results:
    plt.plot(r["xs"], r["Ps"], label=f"M_in = {r['M_in']:.2f}")
plt.xlabel("x")
plt.ylabel("Static Pressure")
plt.legend()
plt.grid(True)
plt.savefig('Final/results/p_profiles.png')

plt.figure()
for r in results:
    plt.plot(r["xs"], r["Ts"], label=f"M_in = {r['M_in']:.2f}")
plt.xlabel("x")
plt.ylabel("Static Temperature")
plt.legend()
plt.grid(True)
plt.savefig('Final/results/t_profiles.png')

plt.figure()
for r in results:
    plt.plot(r["xs"], r["P0s"], label=f"M_in = {r['M_in']:.2f}")
plt.xlabel("x")
plt.ylabel("Stagnation Pressure")
plt.legend()
plt.grid(True)
plt.savefig('Final/results/p0_profiles.png')

plt.figure()
for r in results:
    plt.plot(r["xs"], r["T0s"], label=f"M_in = {r['M_in']:.2f}")
plt.xlabel("x")
plt.ylabel("Stagnation Temperature")
plt.legend()
plt.grid(True)
plt.savefig('Final/results/t0_profiles.png')




