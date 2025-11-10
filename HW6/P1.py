"""
Plot the x-t diagram for a sudden expansion wave. Take a tube whose initial length is
given as 5 meters filled with air at 101325 Pa and 300 K. At time t = 0 one of the endcaps
moves away from the other endcap at a constant speed of 100 m/s. Plot graphically the
characteristics associated with the expansion fan as it emanates from from the sudden
expansion corner in the x-t diagram and subsequently interacts with its reflection coming
from the second stationary endcap.
"""

import numpy as np
import matplotlib.pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0, os.fspath(Path(__file__).parents[1]))

from Tools.misc_functions import get_speed_of_sound
from Tools.constants import *
from HW6.helpers import *  


# Givens
L0 = 5.0     # m
P0 = 101325  # Pa
T0 = 300.0   # K
V  = 100.0   # m/s, right wall speed
N  = 20    # number of characteristics
G  = gamma_air


# Incident fan (left-running C-) from right boundary (x=L0, t=0)
u_incidents = np.linspace(0.0, V, N)
a1 = get_speed_of_sound(T0)
J_plus_0 = get_J(0.0, a1, G, True)  # constant in the simple incident fan

a_incidents = []
Jm_incident = []
for u in u_incidents:
    a = get_a_from_J(J_plus_0, u, G, True)  # from J+ const
    a_incidents.append(a)
    Jm_incident.append(get_J(u, a, G, False))
Jm_incident = np.array(Jm_incident)

# Make matrices (upper triangle valid: c >= r)
shape = (N, N)
x_matrix = np.full(shape, np.nan)
t_matrix = np.full(shape, np.nan)

J_minus_matrix = np.full(shape, np.nan)
J_plus_matrix  = np.full(shape, np.nan)

u_matrix = np.full(shape, np.nan)
a_matrix = np.full(shape, np.nan)

m_plus  = np.full(shape, np.nan)  
m_minus = np.full(shape, np.nan)   

# First row is given by our incident J
for i in range(N):
    J_minus_matrix[0,i] = Jm_incident[i]

# Go through invariants and states node by node 
for r in range(N):                    # reflected C+ index (row)
    for c in range(r, N):             # incident C- index (column)
        if c == r: # diagonal of the matrix is the wall reflection
            # diagonal (wall): seed from incident; u=0, J+ = -J-
            # J_minus_matrix[r, c] = Jm_incident[c]
            J_plus_matrix[r, c]  = -J_minus_matrix[r, c]
            u_matrix[r, c] = 0.0
            a_matrix[r, c] = get_a_from_J(J_plus_matrix[r,c], 0.0, G, True)
        else:
            # --- interior
            # J+, J- come into the node. we should know these already
            Jp_in = J_plus_matrix[r, c-1]
            Jm_in = J_minus_matrix[r, c]

            # # J- comes from:
            # #   row 0: the incident column (simple region)
            # #   rows >=1: the cell above (nonsimple region)
            # if r == 0:
            #     Jm_in = Jm_incident[c]
            # else:
            #     Jm_in = J_minus_matrix[r-1, c]

            # local state from invariants
            a, u = intersect_char(Jp_in, Jm_in, G)

            # write invariants and state at (r,c)
            J_plus_matrix[r, c]  = get_J(u, a, G, True)
            if not (c == N and r == N):
                J_minus_matrix[r+1, c] = get_J(u, a, G, False)
            u_matrix[r, c] = u
            a_matrix[r, c] = a

np.set_printoptions(precision=4, suppress=True, linewidth=120)
print(J_plus_matrix)
print(J_minus_matrix)

# dt/dx slopes after the loop
m_plus  = 1.0 / (u_matrix + a_matrix)   # C+
m_minus = 1.0 / (u_matrix - a_matrix)   # C-


# Now fill in node positions with averaged slopes. also go node by node
for r in range(N):                    # reflected C+ index (row)
    for c in range(r, N):             # incident C- index (column)

        if r ==0: # first row doesn't average the incoming negative slope
            if c == r: # first reflection
                x_matrix[r,c] = 0
                t_matrix[r,c] = m_minus[r,c] * -L0
            else:
                m_minus_avg = m_minus[r,c] 
                m_plus_avg = (m_plus[r, c-1] + m_plus[r,c]) /2
                x_matrix[r,c] = (1/(m_plus_avg-m_minus_avg)) * (m_plus_avg * x_matrix[r,c-1] - m_minus_avg*L0 - t_matrix[r,c-1])
                t_matrix[r,c] = t_matrix[r,c-1] + m_plus_avg * (x_matrix[r,c] - x_matrix[r,c-1])

        # Now move onto other rows, where we do average all slopes
        else:
            if c == r: # diagonal of the matrix is the wall reflection
                x_matrix[r,c] = 0
                m_minus_avg = (m_minus[r,c] + m_minus[r-1,c])/2
                t_matrix[r,c] = -m_minus_avg * x_matrix[r-1,c] + t_matrix[r-1,c]
            else:
                m_minus_avg = (m_minus[r,c] + m_minus[r-1,c])/2
                m_plus_avg = (m_plus[r, c-1] + m_plus[r,c]) /2
                x_matrix[r,c] = (1/(m_plus_avg-m_minus_avg)) * (m_plus_avg * x_matrix[r,c-1] - m_minus_avg*x_matrix[r-1,c] + t_matrix[r-1,c] - t_matrix[r,c-1])
                t_matrix[r,c] = t_matrix[r,c-1] + m_plus_avg * (x_matrix[r,c] - x_matrix[r,c-1])


# Plot results
plt.figure(figsize=(8, 8))
colors = plt.cm.turbo(np.linspace(0, 1, N))  # N distinct colors

# Incident C- characteristics (from L0,0 to each node in first row)
for j in range(N):
    t_end = t_matrix[0, j]
    if not np.isnan(t_end):
        x_vals = [L0, x_matrix[0, j]]
        t_vals = [0.0, t_end]
        plt.plot(x_vals, t_vals, "--", color=colors[j], linewidth=1.2, label="C−" if j == 0 else None)

# Reflected C+ characteristics (from diagonal node to right)
for i in range(N):
    for j in range(i, N - 1):
        if not np.isnan(x_matrix[i, j]) and not np.isnan(x_matrix[i, j+1]):
            x_vals = [x_matrix[i, j], x_matrix[i, j+1]]
            t_vals = [t_matrix[i, j], t_matrix[i, j+1]]
            plt.plot(x_vals, t_vals, "-", color=colors[i], linewidth=1.6, label="C+" if i == 0 and j == i else None)

# Interactions: draw between adjacent rows and same column (C− continuing)
for i in range(N - 1):
    for j in range(i + 1, N):
        if not np.isnan(x_matrix[i, j]) and not np.isnan(x_matrix[i + 1, j]):
            x_vals = [x_matrix[i, j], x_matrix[i + 1, j]]
            t_vals = [t_matrix[i, j], t_matrix[i + 1, j]]
            plt.plot(x_vals, t_vals, "--", color=colors[j], linewidth=1.2)

# Extend final C+ characteristics to right wall
for i in range(N):
    valid_cols = np.where(~np.isnan(x_matrix[i, :]))[0]
    if len(valid_cols) > 0:
        c_last = valid_cols[-1]
        x_last, t_last = x_matrix[i, c_last], t_matrix[i, c_last]
        m_last = m_plus[i, c_last]
        denom = 1.0 - V * m_last
        if abs(denom) > 1e-12:
            t_ext = (t_last + m_last * (L0 - x_last)) / denom
            x_ext = L0 + V * t_ext
            if np.isfinite(t_ext):
                plt.plot([x_last, x_ext], [t_last, t_ext], color=colors[i], linewidth=1.6)

# Moving right wall and fixed left wall
t_max = float(np.nanmax(t_matrix)) * 2
t_wall = np.linspace(0.0, t_max, 200)
x_wall = L0 + V * t_wall
plt.plot(x_wall, t_wall, "k-", linewidth=2.0, label="right wall (moving)")
plt.axvline(0.0, linewidth=1, linestyle=":")
plt.text(0.0, 0, " left wall", rotation=90, va="bottom", ha="left")

# Axes and formatting
plt.xlim(0.0, max(L0, x_wall.max()) * 1.02)
plt.ylim(0.0, t_max)
plt.xlabel("x [m]")
plt.ylabel("t [s]")
plt.title("t-x Diagram: Expansion with suddenly moving wall")
plt.legend(loc="upper left", ncol=2)
plt.tight_layout()
plt.show()
