import numpy as np

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.rayleigh import *
from Tools.numerical_iterator import *
from Tools.constants import *
from Tools.misc_functions import get_speed_of_sound

def solve_combustor_length(M_in, P_in, T_in, m_dot_air, width, m_dot_fuel):
    """
    Calculates the required constant-area tube length using Rayleigh flow 
    and the specific fuel burn time model provided.
    
    Parameters:
    - M_in: Inlet Mach Number
    - P_in: Inlet Static Pressure (Pa)
    - T_in: Inlet Static Temperature (K)
    - m_dot_air: Mass flow of air (kg/s)
    - width: Width into the page (m)
    - m_dot_fuel: Mass flow of fuel (kg/s)
    """
    
    # Constants
    gamma = gamma_air
    R = R_air      # J/(kg*K)
    cp = C_p_air     # J/(kg*K)
    LHV_H2 = 120e6  # 120 MJ/kg = 120,000,000 J/kg

    # Get starting height
    u_in = M_in * get_speed_of_sound(T_in)
    rho_in = P_in / (R * T_in)
    area_in = m_dot_air / (rho_in * u_in)
    height_in = area_in / width

    # ------------------------------------------------
    # INLET STATE (State 1)
    # ------------------------------------------------
    
    # Inlet Isentropic Relations
    # T0 = T * (1 + ((gamma-1)/2) * M^2)
    T0_in = T_in * (1 + ((gamma - 1) / 2) * M_in**2)
    
    # Speed of sound and Velocity
    a_in = np.sqrt(gamma * R * T_in)
    u_in = M_in * a_in
    
    # Calculate Rayleigh critical state (Star quantities)
    # We use the provided T0_T0star function to find the reference T0*
    ratio_T0_in = T0_T0star(M_in, gamma)
    T0_star = T0_in / ratio_T0_in
    
    # Calculate P* for reference (using P_in and provided P_Pstar)
    ratio_P_in = P_Pstar(M_in, gamma)
    P_star = P_in / ratio_P_in

    # ------------------------------------------------
    # HEAT ADDITION
    # ------------------------------------------------
    # Total heat input (Watts)
    Q_total = m_dot_fuel * LHV_H2
    
    # Specific heat input (J/kg) relative to total mass
    q_specific = Q_total / m_dot_air
    
    # Calculate Exit Stagnation Temp (T0_out)
    # q = cp * (T0_out - T0_in)  ->  T0_out = T0_in + q/cp
    T0_out = T0_in + (q_specific / cp)
    
    # Check for Thermal Choking
    # If T0_out > T0_star, we cannot pass that much heat without choking (M=1).
    is_choked = False
    if T0_out > T0_star:
        is_choked = True
        T0_out = T0_star # Limit to max possible
    
    # ------------------------------------------------
    # EXIT STATE (State 2)
    # ------------------------------------------------
    # We need to find M_out such that T0_T0star(M_out) == T0_out / T0_star
    target_ratio = T0_out / T0_star
    
    # Numerical Solver to find M_out
    # Since input is subsonic (M < 1), we look for solution in range [M_in, 1.0]
    
    def func(M):
        return T0_T0star(M, gamma)
    
    if is_choked:
        M_out = 1.0
    else:
    # Solve for M such that T0_T0star(M, gamma) == target_ratio
    # keep the 1e-9 buffer on the upper bound
    # to avoid singularities at M=1.0
        M_out = numerical_iterator(
            func=func(M_out),
            start=M_in,
            end=1.0 - 1e-9,
            goal_y=target_ratio
    )

    # Calculate Exit Static Pressure using provided P_Pstar
    # P_out = P_star * P_Pstar(M_out)
    P_out = P_star * P_Pstar(M_out, gamma)
    
    # Calculate Exit Velocity
    # Need Static Temp T_out first
    # T0 = T * (1 + (gamma-1)/2 * M^2) -> T = T0 / ...
    T_out = T0_out / (1 + ((gamma - 1) / 2) * M_out**2)
    a_out = np.sqrt(gamma * R * T_out)
    u_out = M_out * a_out

    # ------------------------------------------------
    # LENGTH CALCULATION 
    # ------------------------------------------------
    
    # Average Velocity
    u_avg = (u_in + u_out) / 2.0
    
    # Average Conditions for Reaction Rate
    # Use P (atm) and T0 (K). 
    # We approximate these as the average of inlet and outlet.
    P_avg_Pa =  (P_in + P_out) / 2.0
    P_avg_atm = P_avg_Pa / P_sea
    
    T0_avg = (T0_in + T0_out) / 2.0
    
    # Burn Time Calculation
    # Formula: tau = 325 * p^(-1.6) * exp(-0.8 * T0/1000)
    # Result tau is in milliseconds
    tau_ms = 325 * (P_avg_atm**(-1.6)) * np.exp(-0.8 * (T0_avg / 1000.0))
    tau_sec = tau_ms / 1000.0
    
    # Length
    length = u_avg * tau_sec

    return {
        "height_m": height_in,
        "length_m": length,
        "burn_time_ms": tau_ms,
        "is_choked": is_choked,
        "M_in": M_in,
        "M_out": M_out,
        "T0_in": T0_in,
        "T0_out": T0_out,
        "T_in": T_in,
        "T_out": T_out,
        "P_in": P_in,
        "P_out": P_out,
        "U_avg": u_avg
    }

# ==========================================
# EXECUTION
# ==========================================

# Inputs
M_inlet = 0.1
P_inlet = 2754870.323663043
T_inlet = 600.0      # K
m_air = 1.0          # kg/s
m_fuel = 10       # kg/s 
width = 1 # m into page

results = solve_combustor_length(M_inlet, P_inlet, T_inlet, m_air, width, m_fuel)

print("--- RAYLEIGH COMBUSTOR SIZING ---")
if results['is_choked']:
    print("WARNING: Flow is THERMALLY CHOKED. Length calc assumes Q limited to max.")

print(f"Constant Height:   {results['height_m']:.4f} meters")
print(f"Calculated Length:   {results['length_m']:.4f} meters")
print(f"Burn Time:           {results['burn_time_ms']:.4f} ms")
print(f"Avg Velocity:        {results['U_avg']:.2f} m/s")
print(f"Mach Change:         {results['M_in']:.2f} -> {results['M_out']:.2f}")
print(f"Stag. Temp Change:   {results['T0_in']:.1f} K -> {results['T0_out']:.1f} K")
print(f"Static P Change:     {results['P_in']:.1f} Pa -> {results['P_out']:.1f} Pa")
print(f"Static T Change: {results['T_in']:.1f} K -> {results['T_out']:.1f} K")