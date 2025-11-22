"""
Design the 2D steady MOC contour for a supersonic nozzle assuming that the expansion
section is a simple corner of some fixed angle, with the straightening section being
designed to produce uniform outflow. The inflow for your nozzle is to be M = 1, gamma
= 1.4, P = 101325, T = 300 K, incoming flow area of 1m, with 1m depth into the page.
The exit Mach number of your design is to be M = 3. Note to execute this properly you
will need to compute 𝛎 as a function of M, and vice versa.
"""
import numpy as np
import matplotlib.pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0, os.fspath(Path(__file__).parents[1]))

from Tools.constants import *
from HW7.helpers import *  
from Tools.expansion_fan import *


# Givens
G = 1.4 # gamma of air
M_in = 1 # incoming Mach number
P_in = 101325 # Pa, incoming pressure
T_in = 300 # K, incoming temperature
A_in = 1 # m, height of nozzle inflow
D = 1 # m, constant depth of nozzle into page
M_exit = 3 # exit Mach number after the straightening section

N  = 50    # number of characteristics

# Expansion section is the simple corner of some fixed angle, assume this is the form for a minimum length nozzle.
# So then, the expansion happens via a centered Prandtl--Meyer wave emanating from the corner throat
# Get the max angle corresponding to this, then linspace theta from 0 to that angle to create expansion fan
# Note that for this first centered expansion fan without interference, nu = theta
theta_max = nu_func(M_exit, G)/2
theta_incidents = np.linspace(0.0000001, theta_max, N) # radians, not actually 0 start since that'll be undefined
M_incidents = []
K_min_incidents = []
slopes_incidents = []
for theta in theta_incidents:
    M_incidents.append(inverse_nu_func(theta, G))
    K_min_incidents.append(get_K(theta, M_incidents[-1], G, False))
    slopes_incidents.append(get_slope(theta, M_incidents[-1], False))

# Make matrices (upper triangle valid: c >= r)
shape = (N, N)
x_matrix = np.full(shape, np.nan)
y_matrix = np.full(shape, np.nan)

K_minus_matrix = np.full(shape, np.nan)
K_plus_matrix  = np.full(shape, np.nan)

M_matrix = np.full(shape, np.nan)
theta_matrix = np.full(shape, np.nan)

slopes_plus  = np.full(shape, np.nan)  
slopes_minus = np.full(shape, np.nan)   

# First row is given by our incident K
for i in range(N):
    K_minus_matrix[0,i] = K_min_incidents[i]

# Go through invariants and states node by node 
for r in range(N):                    # reflected K+ index (row)
    for c in range(r, N):             # incident K- index (column)
        if c == r: # diagonal of the matrix is the centerline reflection
            # J+ = -J-
            # theta = 0
            K_plus_matrix[r, c]  = -K_minus_matrix[r, c]
            theta_matrix[r, c] = 0.0
            M_matrix[r, c] = get_M_from_K(K_plus_matrix[r,c], 0.0, G, True)
            slopes_plus[r,c] = get_slope(theta_matrix[r, c], M_matrix[r,c], True)
            slopes_minus[r,c] = get_slope(theta_matrix[r, c], M_matrix[r,c], False)
        else:
            # interior
            # K+, K- come into the node. we should know these already
            Kp_in = K_plus_matrix[r, c-1]
            Km_in = K_minus_matrix[r, c]

            # local state from invariants
            theta, M = intersect_char(Kp_in, Km_in, G)

            # write invariants and state at (r,c)
            K_plus_matrix[r, c]  = get_K(theta, M, G, True)
            if not (c == N and r == N):
                K_minus_matrix[r+1, c] = get_K(theta, M, G, False)
            theta_matrix[r, c] = theta
            M_matrix[r, c] = M
            slopes_plus[r,c] = get_slope(theta_matrix[r, c], M_matrix[r,c], True)
            slopes_minus[r,c] = get_slope(theta_matrix[r, c], M_matrix[r,c], False)

# Now fill in node positions with averaged slopes. also go node by node
for r in range(N):                    # reflected C+ index (row)
    for c in range(r, N):             # incident C- index (column)

        if r ==0: # first row doesn't average the incoming negative slope
            if c == r: # first reflection
                y_matrix[r,c] = 0
                x_matrix[r,c] = - (A_in/2) / slopes_minus[r,c]
            else:
                m_minus_avg = slopes_minus[r,c] 
                m_plus_avg = (slopes_plus[r,c] + slopes_plus[r,c-1]) /2
                x_matrix[r,c] = (1/ (m_minus_avg-m_plus_avg)) * ( m_plus_avg * x_matrix[r,c-1] + y_matrix[r, c-1] - A_in/2)
                y_matrix[r,c] = m_minus_avg * x_matrix[r,c] + A_in/2

        # Now move onto other rows, where we do average all slopes
        else:
            if c == r: # diagonal of the matrix is the wall reflection
                y_matrix[r,c] = 0
                m_minus_avg = (slopes_minus[r,c] + slopes_minus[r-1,c])/2
                x_matrix[r,c] = (x_matrix[r-1,c] * m_minus_avg - y_matrix[r-1,c] ) / m_minus_avg
            else:
                m_minus_avg = (slopes_minus[r,c] + slopes_minus[r-1,c])/2
                m_plus_avg = (slopes_plus[r, c-1] + slopes_plus[r,c]) /2
                x_matrix[r,c] = (1/ (m_minus_avg-m_plus_avg)) * ( -m_plus_avg * x_matrix[r,c-1] + m_minus_avg * x_matrix[r-1,c] + y_matrix[r,c-1] - y_matrix[r-1,c])
                y_matrix[r,c] = m_minus_avg * (x_matrix[r,c] - x_matrix[r-1,c]) + y_matrix[r-1,c]


# Intersections to get profile of the top nozzle wall s.t there are no more reflections
x_nozzle_wall = [0.0]
y_nozzle_wall = [A_in / 2.0]
m_start = theta_max
for i in range(N):
    if i == 0:
        x = - slopes_plus[i,N-1] * x_matrix[i,N-1] + y_matrix[i,N-1] - A_in/2
        y = m_start * x + A_in/2
        x_nozzle_wall.append(x)
        y_nozzle_wall.append(y)
    else:
        if i == N-1:
            m_avg = theta_matrix[i,N-1]
        else:
            m_avg = 0.5 * (theta_matrix[i-1,N-1] + theta_matrix[i,N-1])
        x = (1/ (m_avg-slopes_plus[i,N-1])) * ( -slopes_plus[i,N-1] * x_matrix[i,N-1] + y_matrix[i,N-1] - y_nozzle_wall[-1] + m_avg * x_nozzle_wall[-1])
        y = m_avg * (x - x_nozzle_wall[-1]) + y_nozzle_wall[-1]
        x_nozzle_wall.append(x)
        y_nozzle_wall.append(y)

# Plot results
plt.figure(figsize=(8, 6))
colors = plt.cm.turbo(np.linspace(0, 1, N))  # N distinct colors

# Incident C− characteristics (from inlet to first-row nodes)
for j in range(N):
    x_end = x_matrix[0, j]
    y_end = y_matrix[0, j]
    if not np.isnan(x_end) and not np.isnan(y_end):
        x_vals = [0.0, x_end]
        y_vals = [A_in / 2.0, y_end]
        plt.plot(x_vals, y_vals, "--", color=colors[j],
                 linewidth=1.0, label="C−" if j == 0 else None)

# Reflected C+ characteristics (along each row)
for i in range(N):
    for j in range(i, N - 1):
        if (not np.isnan(x_matrix[i, j]) and not np.isnan(y_matrix[i, j]) and
            not np.isnan(x_matrix[i, j + 1]) and not np.isnan(y_matrix[i, j + 1])):
            x_vals = [x_matrix[i, j], x_matrix[i, j + 1]]
            y_vals = [y_matrix[i, j], y_matrix[i, j + 1]]
            plt.plot(x_vals, y_vals, "-", color=colors[i],
                     linewidth=1.0, label="C+" if i == 0 and j == i else None)

# C− interactions between adjacent rows (same column)
for i in range(N - 1):
    for j in range(i + 1, N):
        if (not np.isnan(x_matrix[i, j]) and not np.isnan(y_matrix[i, j]) and
            not np.isnan(x_matrix[i + 1, j]) and not np.isnan(y_matrix[i + 1, j])):
            x_vals = [x_matrix[i, j], x_matrix[i + 1, j]]
            y_vals = [y_matrix[i, j], y_matrix[i + 1, j]]
            plt.plot(x_vals, y_vals, "--", color=colors[j], linewidth=1.0)

# Plot the final C+ characteristic to the nozzle wall
for i in range(N):
    plt.plot( [x_matrix[i, N-1], x_nozzle_wall[ i + 1]],
              [y_matrix[i, N-1], y_nozzle_wall[ i + 1]],
              "-", color=colors[i], linewidth=1.0)

# Plot the nozzle wall
plt.plot(x_nozzle_wall, y_nozzle_wall, "k-", linewidth=2.0, label="Nozzle Wall")

# Inlet and centerline / wall
x_max = np.nanmax(x_matrix)
plt.axvline(0.0, color="k", linewidth=1.0)             # throat plane
plt.axhline(0.0, color="k", linewidth=1.0)             # centerline / lower wall

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title(f"2D MOC Nozzle Design (M_exit = {M_exit}, Number of chars = {N})")
plt.show()

print("Nozzle wall coordinates:")
print(x_nozzle_wall)
print(y_nozzle_wall)

# Save Mach number, x, y matrix for later use in P3
np.save("HW7/data/M_matrix.npy", M_matrix)
np.save("HW7/data/x_matrix.npy", x_matrix)
np.save("HW7/data/y_matrix.npy", y_matrix)
np.save('HW7/data/slopes_plus.npy', slopes_plus)
np.save('HW7/data/slopes_minus.npy', slopes_minus)
