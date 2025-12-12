import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))
from Tools.area_mach import *

# Constants
R = 287
gamma = 1.4

# Converging Section
def design_converging_section(P4, T4, M4, m_dot, length, depth, n_points=100):
    """
    Analyze converging section (Section 4 -> 5) with quasi-1D isentropic flow
    
    Inputs:
        P4: Static pressure at section 4 (Pa)
        T4: Static temperature at section 4 (K)
        M4: Mach number at section 4 (dimensionless)
        m_dot: Mass flow rate (kg/s)
        length: Physical length of converging section (m)
        depth: Physical depth of converging section (m)
        n_points: Number of axial sections
    
    Outputs:
        P, T, M, A, h, x: Arrays with axial distributions
    """
    
    # Stagnation P and T from static
    P04 = P4 * (1 + (gamma-1)/2 * M4**2)**(gamma/(gamma-1))
    T04 = T4 * (1 + (gamma-1)/2 * M4**2)

    # Other static properties at section 4
    rho4 = P4 / (R * T4)
    a4 = np.sqrt(gamma * R * T4)
    u4 = M4 * a4
    
    # Calculate area at section 4 from mass flow rate
    A4 = m_dot / (rho4 * u4)

    # Calculate throat area (A*)
    A_ratio_sq_4 = area_mach_relation(M4, gamma)
    A_star = A4 / np.sqrt(A_ratio_sq_4)
    
    # Section 5 is the throat
    M5 = 1.0
    A5 = A_star
    
    # Stagnation properties (conserved for isentropic flow)
    P05 = P04
    T05 = T04
    
    # Static properties at Section 5
    T5 = T05 / (1 + (gamma-1)/2 * M5**2)
    P5 = P05 / ((1 + (gamma-1)/2 * M5**2)**(gamma/(gamma-1)))
    rho5 = P5 / (R * T5)
    a5 = np.sqrt(gamma * R * T5)
    u5 = M5 * a5

    # Mass flow rate sanity check
    m_dot_5 = rho5 * u5 * A5
    if not np.isclose(m_dot, m_dot_5, rtol=1e-6):
        raise ValueError(f"Mass flow rate not conserved: input={m_dot}, calculated={m_dot_5}")
    
    # Contour geometry
    x = np.linspace(0, length, n_points)
    xi = x / length
    A = A4 - (A4 - A5) * (3*xi**2 - 2*xi**3)    # Cubic polynomial makes sure dA/dx = 0 at boundaries
    
    # Since the converging section is treated as isentropic flow, the stagnation pressure and temp are conserved
    # regardless of the actual contour. The exit conditions also depend only on the inlet properties and exit 
    # mach M5 = 1. Therefore the contour doesn't actually matter. However, we use a cubic polynomial to make sure the
    # the transitions between sections are smooth which would help minimize any flow separation and pressure losses
    # that would occur from sharp turns in the real world.

    # Calculate flow properties at each axial section
    A_over_Astar = A / A_star

    # Find Mach number at each location
    M_values = []
    for area_ratio in A_over_Astar:
        A_ratio_sq = area_ratio**2
        M_solutions = inverse_area_mach_relation(A_ratio_sq, gamma)
        M_subsonic = M_solutions[0]  # Take subsonic solution
        M_values.append(M_subsonic)

    M = np.array(M_values)

    # Calculate pressure and temperature ratios
    p_over_p0 = 1/((1 + (gamma-1)/2 * M**2)**(gamma/(gamma-1)))
    T_over_T0 = 1/(1 + (gamma-1)/2 * M**2)

    # Calculate static properties
    P = p_over_p0 * P04
    T = T_over_T0 * T04

    # Calculate height
    h = A / (depth)

    return P, T, M, m_dot, A, h, x

def analyze_converging_section(hs, P4, T4, M4, depth):
    """
    Analyze converging section (Section 4 -> 5) with quasi-1D isentropic flow
    
    Inputs:
        hs: Height of converging section (m)
        P4: Static pressure at section 4 (Pa)
        T4: Static temperature at section 4 (K)
        M4: Mach number at section 4 (dimensionless)
        depth: Physical depth of converging section (m)

    Outputs:
        Ps, Ts, Ms: Arrays with axial distributions
    """

    # Stagnation P and T from static
    P04 = P4 * (1 + (gamma-1)/2 * M4**2)**(gamma/(gamma-1))
    T04 = T4 * (1 + (gamma-1)/2 * M4**2)
    
    # Calculate areas
    As = hs * depth

    # Calculate throat area (A*)
    A_ratio_sq_4 = area_mach_relation(M4, gamma)
    A4 = As[0]
    A_star = A4 / np.sqrt(A_ratio_sq_4)

    # Calculate flow properties at each axial section
    A_over_Astar = As / A_star

    # Find Mach number at each location
    M_values = []
    for area_ratio in A_over_Astar:
        A_ratio_sq = area_ratio**2
        M_solutions = inverse_area_mach_relation(A_ratio_sq, gamma)
        M_subsonic = M_solutions[0]  # Take subsonic solution
        if np.isnan(M_subsonic):
            M_values.append(M_solutions[1])
        else:
            M_values.append(M_subsonic)

    Ms = np.array(M_values)

    # Calculate pressure and temperature ratios
    p_over_p0 = 1/((1 + (gamma-1)/2 * Ms**2)**(gamma/(gamma-1)))
    T_over_T0 = 1/(1 + (gamma-1)/2 * Ms**2)

    # Calculate static properties
    Ps = p_over_p0 * P04
    Ts = T_over_T0 * T04

    return Ps, Ts, Ms

# Run
if __name__ == "__main__":
    P4 = 200000  # Pa
    T4 = 2000    # K
    M4 = 0.3      # dimensionless
    m_dot = 5.0  # kg/s
    length = 1.0  # m
    depth = 1.0  # m

    P, T, M, m_dot, A, h, x = design_converging_section(P4, T4, M4, m_dot, length, depth, n_points=100)

    # Section 5 properties
    print(f"P5 = {P[-1]} Pa")
    print(f"T5 = {T[-1]} K")
    print(f"M5 = {M[-1]}")
    print(f"mdot = {m_dot} kg/s")
    print(f"A5 = {A[-1]} m^2")
    print(f"h5 = {h[-1]} m")

    # Plots
    # Geometry
    plt.figure()
    plt.plot(x, -h, 'b-', linewidth=2)
    plt.plot(x, np.zeros(len(x)), 'b-', linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('h (m)')
    plt.title('Converging Section Geometry')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # Area
    plt.figure()
    plt.plot(x, A, linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('A (m^2)')
    plt.title('Area Distribution')
    plt.grid(True)
    plt.show()

    # Pressure
    plt.figure()
    plt.plot(x, P, linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('P (Pa)')
    plt.title('Pressure Distribution')
    plt.grid(True)
    plt.show()

    # Temperature
    plt.figure()
    plt.plot(x, T, linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('T (K)')
    plt.title('Temperature Distribution')
    plt.grid(True)
    plt.show()

    # Mach number
    plt.figure()
    plt.plot(x, M, linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('M')
    plt.title('Mach Number Distribution')
    plt.grid(True)
    plt.show()

    # Off-Design Quasi 1-D Analysis
    P4 = 100000  # Pa
    T4 = 1000    # K
    M4 = 0.3      # dimensionless

    P5, T5, M5 = analyze_converging_section(h, P4, T4, M4, depth)
    
    print(f"P5 = {P5[-1]} Pa")
    print(f"T5 = {T5[-1]} K")
    print(f"M5 = {M5[-1]}")