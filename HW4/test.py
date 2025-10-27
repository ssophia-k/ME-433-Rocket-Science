
# ---- Replace inner part starting where you build lifts/drags ----
# assume oblique_shock takes (M, theta_deg) and returns (M2, Pr=P2/P1, Tr, beta_deg)
# assume expansion_wave(M, theta_deg) returns new Mach (M2)
# assume stagnation_pressure(M) returns p0/p (or p0_over_p). Adjust usage below if its definition differs.

import numpy as np
from matplotlib import pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.oblique_shock import *
from Tools.expansion_fan import *
from Tools.isentropic import P0_P
from Tools.misc_functions import get_speed_of_sound
from Tools.constants import *
from HW4.P3_helpers import get_surface_angles, get_turn_angles

M0 = 3
gamma = 1.4
P0r = P0_P(M0, gamma)

for airfoil_id in range(1, 5):
    with open(f"HW4/airfoils/airfoil{airfoil_id}.txt", "r") as file:
        xs = [float(i) for i in file.readline().strip().split(",")]
        ys = [float(i) for i in file.readline().strip().split(",")]
        xs.insert(0, -1.0); ys.insert(0, 0.0)
        xs.append(2.0);    ys.append(0.0)

    lifts = []
    drags = []
    alphas = np.linspace(-5, 5, 1000)  # specify number of points you want
    for alpha_deg in alphas:
        alpha_rad = np.deg2rad(alpha_deg)

        # Precompute panel orientations (radians) for the polygon panels
        # panels run between nodes j -> j+1
        panel_thetas = []
        for j in range(len(xs)-1):
            dx = xs[j+1] - xs[j]
            dy = ys[j+1] - ys[j]
            panel_thetas.append(np.arctan2(dy, dx))
        # We'll be using panels inside loop over panels i from 0..Npan-1
        Npan = len(panel_thetas) - 2   # you were iterating len(xs)-3; adjust if needed
                                       # but below we will iterate panels 0..Npan-1

        # initialize top/bottom pressure and Mach lists
        P_top = []
        P_bottom = []
        M_top = []
        M_bottom = []

        # For the first panel, incoming flow is freestream direction = alpha_rad (x-axis rotated)
        incoming_top_angle = alpha_rad  # angle of flow approaching top first panel
        incoming_bottom_angle = alpha_rad  # similarly for bottom

        # We'll compute force contributions (global axes) and then sum
        total_lift = 0.0
        total_drag = 0.0

        # iterate panels (use same indexing convention as earlier: i loops over structural panels)
        # choose i range consistent with your geometry; here I use 0..(len(xs)-3) to match original
        for i in range(len(xs) - 3):
            # panel between node i+1 and i+2 (this matches original code's segment_length calc)
            dx = xs[i+2] - xs[i+1]
            dy = ys[i+2] - ys[i+1]
            panel_theta = np.arctan2(dy, dx)   # radians

            # turning angles (signed) in radians: how much the flow turns across this panel
            # positive means CCW (deflection that would cause an oblique shock on top if supersonic)
            turn_top_rad = panel_theta - incoming_top_angle
            turn_bot_rad = incoming_bottom_angle - panel_theta  # bottom sees opposite sign

            # convert to degrees for your oblique_shock/expansion_wave if they expect degrees
            turn_top_deg = np.rad2deg(turn_top_rad)
            turn_bot_deg = np.rad2deg(turn_bot_rad)

            # ---- Top surface update ----
            if i == 0:
                # first panel: incoming Mach is M0
                M_in_top = M0
                P_in_top = 1.0  # use nondimensional static pressure base: choose P_ref=1 at freestream surface
                # If you used P0r earlier, careful: here we treat P_in relative to freestream static pressure
            else:
                M_in_top = M_top[-1]
                P_in_top = P_top[-1]

            if turn_top_deg > 0:
                # oblique shock: returns M2 and Pr = P2/P1
                _, Pr_top, _,  M2_top = mach_function(M_in_top, gamma, turn_top_deg)
                P_out_top = P_in_top * Pr_top
            elif turn_top_deg < 0:
                # expansion: returns new Mach M2
                M2_top = get_M2_from_nu(M_in_top, gamma, -turn_top_deg)
                # convert using stagnation_pressure if necessary
                # if stagnation_pressure(M) returns p0/p, then static p scales like: p_out = p_in * ( p0/p(M_out) / p0/p(M_in) )
                p0_over_p_in = P0_P(M_in_top, gamma)
                p0_over_p_out = P0_P(M2_top, gamma)
                P_out_top = P_in_top * (p0_over_p_in / p0_over_p_out)
            else:
                raise ValueError("zero turning angle encountered")

            # push top results
            M_top.append(M2_top)
            P_top.append(P_out_top)

            # ---- Bottom surface update ----
            if i == 0:
                M_in_bot = M0
                P_in_bot = 1.0
            else:
                M_in_bot = M_bottom[-1]
                P_in_bot = P_bottom[-1]

            if turn_bot_deg > 0:
                _, Pr_bot, _,  M2_bot = mach_function(M_in_bot, gamma, turn_bot_deg)
                P_out_bot = P_in_bot * Pr_bot
            elif turn_bot_deg < 0:
                M2_bot = get_M2_from_nu(M_in_bot, gamma, -turn_bot_deg)
                p0_over_p_in = P0_P(M_in_bot, gamma)
                p0_over_p_out = P0_P(M2_bot, gamma)
                P_out_bot = P_in_bot * (p0_over_p_in / p0_over_p_out)
            else:
                raise ValueError("zero turning angle encountered on bottom")

            M_bottom.append(M2_bot)
            P_bottom.append(P_out_bot)

            # ---- compute panel force contribution ----
            # Pressure difference (bottom - top) acts normal to panel; project to x,y:
            dp = P_out_bot - P_out_top
            section_length = np.hypot(dx, dy)
            # force per panel in global x,y (pressure acts normal to panel: normal = [ -sin(theta), cos(theta) ] )
            nx = -np.sin(panel_theta)
            ny =  np.cos(panel_theta)
            Fx = dp * section_length * nx   # x-component (drag is along +x freestream)
            Fy = dp * section_length * ny   # y-component (lift is +y)

            # rotate global force into aerodynamic frame if freestream not aligned with +x:
            # our Fx,Fy are already in global inertial where freestream is at angle alpha_rad relative to x-axis
            # But if we defined freestream direction by alpha, drag is component along freestream vector:
            # drag = Fx*cos(alpha) + Fy*sin(alpha)
            # lift = -Fx*sin(alpha) + Fy*cos(alpha)   (positive lift upwards)
            drag_panel = Fx * np.cos(alpha_rad) + Fy * np.sin(alpha_rad)
            lift_panel = -Fx * np.sin(alpha_rad) + Fy * np.cos(alpha_rad)

            total_drag += drag_panel
            total_lift += lift_panel

            # update incoming flow direction for next panel:
            # After this panel, the flow direction changes by +turn_top_rad for top path
            # and -turn_bot_rad for bottom path; the new incoming angles for next panel are:
            # (we assume small deflections add up)
            incoming_top_angle = incoming_top_angle + turn_top_rad
            incoming_bottom_angle = incoming_bottom_angle - turn_bot_rad

        # End panel loop; append coefficients (normalize as needed)
        # If you want nondimensional C_l, C_d, divide by dynamic pressure*chord etc. Here we just store totals.
        lifts.append(total_lift)
        drags.append(total_drag)

    # plot for this airfoil
    plt.plot(drags, lifts, label=f"airfoil {airfoil_id}")
# end airfoil loop
plt.xlabel("C_d")
plt.ylabel("C_l")
plt.title("C_l vs C_d for alpha from -5 to 5 degrees")
plt.legend()
plt.grid()
plt.show()
