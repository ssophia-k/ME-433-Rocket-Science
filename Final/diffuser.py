import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.constants import *
from Tools.area_mach import *
from Tools.isentropic import *
from Tools.misc_functions import get_speed_of_sound

# Diffuser Section, to be treated as a quasi 1D isentropic flow (2 to 3)

# Input m-dot, M, P, T
# Output M, P, T, diffuser shape
# Want perfect isentropic compression as to avoid shocks, waves on the interior of the engine

# Flow should be entering subsonic (due to the normal shock at the inlet)

def find_diffuser(M_in, P_in, T_in, m_dot, width, M_exit, Length, Resolution):
    gamma = gamma_air
    R = R_air

    # --- Calculation ---

    # Get starting height
    u_in = M_in * get_speed_of_sound(T_in)
    rho_in = P_in / (R * T_in)
    area_in = m_dot / (rho_in * u_in)
    height_in = area_in / width

    # Establish Stagnation Properties (Constant throughout the diffuser)
    P0 = get_P0_from_static(P_in, M_in, gamma)
    T0 = get_T0_from_static(T_in, M_in, gamma)
    A_star = area_in/ (np.sqrt(area_mach_relation(M_in, gamma)))

    print(f"--- Design Parameters ---")
    print(f"Total Pressure (P0): {P0/1000:.2f} kPa")
    print(f"Theoretical A*:      {A_star:.4f} m^2")

    # Generate Profile
    # We define a linear deceleration from M_in to M_exit over the Length L.
    # (You could change this to linear Area change if preferred, but linear Mach is smoother for flow)
    x_coords = np.linspace(0, Length, Resolution)
    mach_dist = np.linspace(M_in, M_exit, Resolution)

    results = []

    for x, M in zip(x_coords, mach_dist):
        # 1. Calculate Required Area for this Mach
        area = get_area_from_mach(M, A_star, gamma)
        
        # 2. Calculate height (y coordinate)
        height = area / width
        
        # 3. Calculate Static Pressure
        # P_static = P0 / (P0/P ratio)
        p_static = P0 / P0_P(M, gamma)
        
        # 4. Calculate Static Temperature
        T_static = T0 / T0_T(M, gamma)

        results.append([x, height, area, M, p_static, T_static])

    # Convert to DataFrame
    df = pd.DataFrame(results, columns=['x', 'y', 'area', 'Mach', 'Pressure', 'Temperature'])

    return df


# --- Design Inputs ---

# INLET CONDITIONS (From the inlet portion)
M_in = 0.7      # Entering Mach (Must be < 1)
P_in = 2000000     # Inlet Static Pressure (Pa)
T_in = 716       # Inlet Static Temp (K)
m_dot = 100.0       # Mass flow (kg/s)
width = 1 # width into the page (m)

# DESIGN GOALS
M_exit = 0.1       # Target Mach at combustor face
Length = 0.8      # Length of the diffuser section (meters)
Resolution = 100   # Number of coordinate points

df = find_diffuser(M_in, P_in, T_in, m_dot, width, M_exit, Length, Resolution)

# --- Outputs ---

print(f"\n--- Diffuser Geometry ---")
print(f"Inlet height:  {df['y'].iloc[0]:.2f} m")
print(f"Exit height:   {df['y'].iloc[-1]:.2f} m")
print(f"Area Expansion Ratio: {df['area'].iloc[-1]/df['area'].iloc[0]:.2f}")
print(f"Pressure Rise: {(df['Pressure'].iloc[-1] - P_in):.2f} Pa")
print(f"Pressure: {(df['Pressure'].iloc[-1])} Pa")
print(f"Temperature: {(df['Temperature'].iloc[-1])} K")

# Plotting
plt.figure(figsize=(10, 8))

# Profile Plot
plt.subplot(2, 1, 1)
plt.plot(x_coords, [0] * Resolution, 'k-', linewidth=2, label='Wall')
plt.plot(df['x'], -df['y'], 'k-', linewidth=2)
plt.fill_between(df['x'], [0] * Resolution, -df['y'], color='lightblue', alpha=0.3)
plt.title(f'Diffuser Profile (Decelerating M={M_in} to M={M_exit})')
plt.ylabel('Height (m)')
plt.legend()
plt.axis('equal')
plt.grid(True)

# Pressure Plot
plt.subplot(2, 1, 2)
plt.plot(df['x'], df['Pressure']/1000, 'r-', linewidth=2)
plt.xlabel('Length (m)')
plt.ylabel('Static Pressure (kPa)')
plt.title('Static Pressure Recovery')
plt.grid(True)

plt.tight_layout()
plt.show()
