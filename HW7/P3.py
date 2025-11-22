"""
Compare the Quasi 1-D profile with the 2D MOC profile by constructing an average
Mach number, Pressure, and temperature for each location along the distance of the
nozzle. To construct these averages the question becomes how do you average? For the
purposes of comparison, construct an area averaged representation and mass flux (𝞺u)
averaged representation and compare both to the quasi 1D results, comment on the
results.
"""

import numpy as np
import matplotlib.pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0, os.fspath(Path(__file__).parents[1]))

from Tools.constants import *
from HW7.helpers import *  
from Tools.expansion_fan import *
from Tools.isentropic import *
from Tools.misc_functions import get_speed_of_sound
from HW7.P3_helpers import *

# Givens
G = 1.4 # gamma of air
M_in = 1 # incoming Mach number
P_in = 101325 # Pa, incoming pressure
T_in = 300 # K, incoming temperature
A_in = 1 # m, height of nozzle inflow
D = 1 # m, constant depth of nozzle into page
M_exit = 3 # exit Mach number after the straightening section

N = 50    # number of characteristics

# From Problem 1:
x_nozzle_wall = np.load('HW7/data/x_nozzle_wall.npy')
y_nozzle_wall = np.load('HW7/data/y_nozzle_wall.npy')
M_matrix = np.load("HW7/data/M_matrix.npy")
x_matrix = np.load("HW7/data/x_matrix.npy")
y_matrix = np.load("HW7/data/y_matrix.npy")
slopes_plus = np.load("HW7/data/slopes_plus.npy")
slopes_minus = np.load("HW7/data/slopes_minus.npy")

M_cell_matrix = np.full(M_matrix.shape, np.nan)
for r in range(N): 
    if r == 0:
        min_c = 0
    else:
        min_c = r-1                   
    for c in range(min_c, N):    
        if r == 0:
            if c == N-1:
                M_cell_matrix[r, c] = M_matrix[r, c]
            else:
                M_cell_matrix[r, c] = (M_matrix[r,c] + M_matrix[r,c+1]) / 2.0
        else:     
            if c == r-1:
                M_cell_matrix[r, c] = (M_matrix[r-1,c] + M_matrix[r-1,c+1] + M_matrix[r,c+1]) / 3.0
            elif c == N-1:
                M_cell_matrix[r, c] = (M_matrix[r-1,c] + M_matrix[r,c]) / 2.0
            else:
                M_cell_matrix[r,c] = (M_matrix[r-1,c] + M_matrix[r-1,c+1] + M_matrix[r,c] + M_matrix[r,c+1]) / 4.0




# Given we know the behavior and definitions of our cell matrices, we can classify any point by checking the matrix
def return_M(x_p, y_p):
    # first check M_in and M_e
    if point_in_triangle( (x_p, y_p),
                            (0.0, 0.0), 
                            (0.0, 0.5), 
                            (x_matrix[0,0], y_matrix[0,0]) ):
            return M_in
    elif point_in_triangle( (x_p, y_p),
                            (x_nozzle_wall[-1], 0.0), 
                            (x_nozzle_wall[-1], y_nozzle_wall[-1]), 
                            (x_matrix[N-1,N-1], y_matrix[N-1,N-1]) ):
            return M_exit
    else: 
        for r in range(N): 
            if r == 0:
                min_c = 0
            else:
                min_c = r-1                   
            for c in range(min_c, N):    
                if r == 0: # check incident cells on left

                    if c == N-1: # check top left corner
                        # right bound
                        x_rb = (y_p - y_matrix[r,c] + slopes_plus[r,c] * x_matrix[r,c] ) / slopes_plus[r,c]

                        # lower bound will depend
                        if x_p > x_matrix[r,c]:
                            y_lb = y_matrix[r,c] + slopes_plus[r,c] * (x_p - x_matrix[r,c])
                        elif x_p == x_matrix[r,c]:
                            y_lb = y_matrix[r,c]
                        else:
                            y_lb = y_matrix[r,c] + slopes_minus[r,c] * (x_p - x_matrix[r,c])

                        if y_p >= y_lb and x_p <= x_rb:
                            return M_cell_matrix[r,c]
                        
                    else: # check other cells on the left hand side
                        # right bound
                        x_rb = (y_p - y_matrix[r,c] + slopes_plus[r,c] * x_matrix[r,c] ) / slopes_plus[r,c]

                        # left bound
                        x_lb = (y_p - y_matrix[r,c] + slopes_minus[r,c] * x_matrix[r,c]) / slopes_minus[r,c]

                        # lower bound will depend
                        if x_p > x_matrix[r,c]:
                            y_lb = y_matrix[r,c] + slopes_plus[r,c] * (x_p - x_matrix[r,c])
                        elif x_p == x_matrix[r,c]:
                            y_lb = y_matrix[r,c]
                        else:
                            y_lb = y_matrix[r,c] + slopes_minus[r,c] * (x_p - x_matrix[r,c])

                        # upper bound
                        y_ub = y_matrix[r,c+1] + slopes_minus[r,c+1] * (x_p - x_matrix[r,c+1])

                        if y_p >= y_lb and x_p >= x_lb and x_p <= x_rb and y_p <= y_ub:
                            return M_cell_matrix[r,c]
                        
                else:     
                    if c == r-1: # bottom triangular regions along the midline
                        if y_p == 0.0:
                            if x_p >= x_matrix[r-1,c] and x_p <= x_matrix[r, c+1]:
                                return (M_matrix[r-1,c] + M_matrix[r, c+1])/1
                        elif point_in_triangle( (x_p, y_p), 
                                            (x_matrix[r-1,c], y_matrix[r-1,c]), 
                                            (x_matrix[r-1,c+1], y_matrix[r-1,c+1]), 
                                            (x_matrix[r,c+1], y_matrix[r,c+1]) ):
                            return M_cell_matrix[r,c]
                        
                    elif c == N-1: # top sections until wall
                        if y_p <= y_matrix[r-1,c]: # bottom half triangle
                            right_intersect_y = y_matrix[r-1,c]
                            right_intersect_x = ( right_intersect_y - y_matrix[r,c] + slopes_plus[r,c] * x_matrix[r,c]) / slopes_plus[r,c]
                            if point_in_triangle( (x_p, y_p),
                                                    (x_matrix[r-1,c], y_matrix[r-1,c]), 
                                                    (x_matrix[r,c], y_matrix[r,c]), 
                                                    (right_intersect_x, right_intersect_y) ):
                                    return M_cell_matrix[r,c]
                        else: # on top of triangle
                            y_lb = y_matrix[r-1,c]
                            x_lb = (y_p - y_matrix[r-1,c] + slopes_plus[r-1,c] * x_matrix[r-1,c]) / slopes_plus[r-1,c]
                            x_rb = (y_p - y_matrix[r,c] + slopes_plus[r,c] * x_matrix[r,c]) / slopes_plus[r,c]
                            if y_p >= y_lb and x_p >= x_lb and x_p <= x_rb:
                                return M_cell_matrix[r,c]
                        
                    else: # stuff in the middle
                        # Case 1, left node higher than right node
                        if y_matrix[r-1,c] >= y_matrix[r,c+1]:
                            if y_p <= y_matrix[r,c+1]: #bottom half triangle of quadrilateral
                                m_minus = (slopes_minus[r-1,c] + slopes_minus[r,c])/2
                                left_intersect_y = y_matrix[r,c+1]
                                left_intersect_x = (left_intersect_y - y_matrix[r,c] + m_minus * x_matrix[r,c]) / m_minus
                                if point_in_triangle( (x_p, y_p),
                                                        (x_matrix[r,c+1], y_matrix[r,c+1]), 
                                                        (x_matrix[r,c], y_matrix[r,c]), 
                                                        (left_intersect_x, left_intersect_y) ):
                                        return M_cell_matrix[r,c]
                            elif y_p >= y_matrix[r-1,c]: # top half triangle of quadrilateral
                                m_minus = (slopes_minus[r-1,c+1] + slopes_minus[r,c+1])/2
                                right_intersect_y = y_matrix[r-1,c]
                                right_intersect_x = ( right_intersect_y - y_matrix[r-1,c+1] + m_minus * x_matrix[r-1,c+1]) / m_minus
                                if point_in_triangle( (x_p, y_p),
                                                        (x_matrix[r-1,c], y_matrix[r-1,c]), 
                                                        (x_matrix[r-1,c+1], y_matrix[r-1,c+1]), 
                                                        (right_intersect_x, right_intersect_y) ):
                                        return M_cell_matrix[r,c]
                            else: # little strip in the middle
                                y_ub = y_matrix[r-1,c]
                                y_lb = y_matrix[r,c+1]
                                m_minus_left = (slopes_minus[r-1,c] + slopes_minus[r,c])/2
                                m_minus_right = (slopes_minus[r-1,c+1] + slopes_minus[r,c+1])/2
                                x_lb = (y_p - y_matrix[r,c] + m_minus_left * x_matrix[r,c]) / m_minus_left
                                x_rb = (y_p - y_matrix[r,c+1] + m_minus_right * x_matrix[r,c+1]) / m_minus_right
                                if y_p >= y_lb and y_p <= y_ub and x_p >= x_lb and x_p <= x_rb:
                                    return M_cell_matrix[r,c]

                        # Case 2, right node higher than left node
                        else: 
                            if y_p <= y_matrix[r-1,c]: # bottom half triangle of quadrilateral
                                m_plus = (slopes_plus[r,c] + slopes_plus[r,c+1])/2
                                right_intersect_y = y_matrix[r-1,c]
                                right_intersect_x = ( right_intersect_y - y_matrix[r,c] + m_plus * x_matrix[r,c]) / m_plus
                                if point_in_triangle( (x_p, y_p),
                                                        (x_matrix[r-1,c], y_matrix[r-1,c]), 
                                                        (x_matrix[r,c], y_matrix[r,c]), 
                                                        (right_intersect_x, right_intersect_y) ):
                                        return M_cell_matrix[r,c]
                            elif y_p >= y_matrix[r,c+1]: # top half triangle of quadrilateral
                                m_plus = (slopes_plus[r-1,c] + slopes_plus[r-1,c+1])/2
                                left_intersect_y = y_matrix[r,c+1]
                                left_intersect_x = ( left_intersect_y - y_matrix[r-1,c+1] + m_plus * x_matrix[r-1,c+1]) / m_plus
                                if point_in_triangle( (x_p, y_p),
                                                        (x_matrix[r,c+1], y_matrix[r,c+1]), 
                                                        (x_matrix[r-1,c+1], y_matrix[r-1,c+1]), 
                                                        (left_intersect_x, left_intersect_y) ):
                                        return M_cell_matrix[r,c]
                            else: # little strip in the middle
                                y_ub = y_matrix[r,c+1]
                                y_lb = y_matrix[r-1,c]
                                m_plus_left = (slopes_plus[r-1,c] + slopes_plus[r-1,c+1])/2
                                m_plus_right = (slopes_plus[r,c] + slopes_plus[r,c+1])/2

                                x_lb = (y_p - y_matrix[r-1,c] + m_plus_left * x_matrix[r-1,c]) / m_plus_left
                                x_rb = (y_p - y_matrix[r,c] + m_plus_right * x_matrix[r,c]) / m_plus_right
                                if y_p >= y_lb and y_p <= y_ub and x_p >= x_lb and x_p <= x_rb:
                                    return M_cell_matrix[r,c]
    print(f"Warning: No M found for point ({x_p:.3f}, {y_p:.3f})")
    return None
                
# now make linspace to test slices of x upwards along the nozzle
x_vals = np.linspace(0, x_nozzle_wall[-1], 300)

M_area_avg = []
M_massflux_avg = []

P_area_avg = []
P_massflux_avg = []

T_area_avg = []
T_massflux_avg = []

# We can also get non-normalized P, T, rho based off finding stagnation at the starting conditions
P0 = P_in * P0_P(M_in, G)   
T0 = T_in * T0_T(M_in, G)
rho0 = P0 / (R_air * T0)
rho_in = P_in / (R_air * T_in)
u_in = get_speed_of_sound(T_in) * M_in
mass_flux_0 = rho_in * u_in * 1

# march along to get area and mass flux averages
for x_p in x_vals:
    # Determine nozzle wall height at this x-location
    y_wall = np.interp(x_p, x_nozzle_wall, y_nozzle_wall)
    
    # Discretize y from 0 to y_wall
    num_y = 200
    y_vals = np.linspace(0.0, y_wall, num_y)
    dy = y_wall / num_y  # uniform spacing

    # Accumulators
    M_area = 0
    M_mf = 0
    P_area = 0
    P_mf = 0
    T_area = 0
    T_mf = 0
    mass_flux_total = 0

    for y_p in y_vals:
        M = return_M(x_p, y_p)
        if M is None:
            print(f"Warning: No M seen for point ({x_p:.3f}, {y_p:.3f})")
            continue
        if np.isnan(M):
            print(f"Warning: M is NaN at ({x_p:.3f}, {y_p:.3f})")
            continue
        elif np.isposinf(M) or np.isneginf(M):
            print(f"Warning: M is inf at ({x_p:.3f}, {y_p:.3f})")
            continue

        T = T0 / T0_T(M, G)
        P = P0 / P0_P(M, G)
        rho = rho0 / rho0_rho(M, G)
        a = get_speed_of_sound(T)
        u = M * a
        mf = rho * u  # local mass flux

        # Area-averaged quantities (integrate then normalize)
        M_area += M * dy
        P_area += P * dy
        T_area += T * dy

        # Mass-flux-weighted integration
        mass_flux_total += mf * dy
        M_mf += M * mf * dy
        P_mf += P * mf * dy
        T_mf += T * mf * dy

    # Normalize and append
    M_area_avg.append(M_area / y_wall)
    P_area_avg.append(P_area / y_wall)
    T_area_avg.append(T_area / y_wall)

    # M_massflux_avg.append(M_mf / mass_flux_0)
    # P_massflux_avg.append(P_mf / mass_flux_0)
    # T_massflux_avg.append(T_mf / mass_flux_0)

    # if mass_flux_total == 0:
    #     M_massflux_avg.append(M_area / y_wall)
    #     P_massflux_avg.append(P_area / y_wall)
    #     T_massflux_avg.append(T_area / y_wall)
    # else:
    M_massflux_avg.append(M_mf / mass_flux_total)
    P_massflux_avg.append(P_mf / mass_flux_total)
    T_massflux_avg.append(T_mf / mass_flux_total)




# bring in our M, P, T from earlier in P2
P2_Mach_list = np.load('HW7/data/P2_mach_list.npy')
P2_P_list = np.load('HW7/data/P2_P_list.npy')
P2_T_list = np.load('HW7/data/P2_T_list.npy')

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

# Mach number
axes[0].plot(x_nozzle_wall, P2_Mach_list, label="Quasi-1D Flow")
axes[0].plot(x_vals, M_area_avg, label="Area-Averaged")
axes[0].plot(x_vals, M_massflux_avg, label="Mass-Flux-Averaged")
axes[0].set_ylabel('Mach number, M')
axes[0].set_title('Comparison of Quasi-1D, Area-Averaged, and Mass-Flux-Averaged Flow Properties in MOC Designed Nozzle')
axes[0].grid(True)

# Pressure 
axes[1].plot(x_nozzle_wall, P2_P_list, label="Quasi-1D Flow")
axes[1].plot(x_vals, P_area_avg, label="Area-Averaged")
axes[1].plot(x_vals, P_massflux_avg, label="Mass-Flux-Averaged")
axes[1].set_ylabel('Pressure [Pa]')
axes[1].grid(True)

# Temperature
axes[2].plot(x_nozzle_wall, P2_T_list, label="Quasi-1D Flow")
axes[2].plot(x_vals, T_area_avg, label="Area-Averaged")
axes[2].plot(x_vals, T_massflux_avg, label="Mass-Flux-Averaged")
axes[2].set_xlabel('x location [m]')
axes[2].set_ylabel('Temperature [K]')
axes[2].grid(True)

plt.tight_layout()
plt.legend()
plt.show()
