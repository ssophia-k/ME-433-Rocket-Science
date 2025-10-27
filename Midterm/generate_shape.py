import numpy as np
import matplotlib.pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0, os.fspath(Path(__file__).parents[1]))

from Tools.oblique_shock import *
from Tools.expansion_fan import *
from Tools.isentropic import P0_P
from Tools.misc_functions import get_speed_of_sound
from HW4.P3_helpers import get_surface_angles, get_turn_angles
from Tools.constants import *
from Midterm.cost import cost_function

tol = 1e-8

def get_Mach_numbers_p_ratios(x, y, top, M_in, gamma):
    Ms = [M_in]
    P_ratio = []
    thetas = get_turn_angles(x, y, aoa=0, top=top)

    for theta in thetas:
        if np.isnan(Ms[-1]):
            Ms.append(np.nan); P_ratio.append(np.nan); continue

        if theta > 0.0:
            if Ms[-1] <= 1.0:
                Ms.append(np.nan); P_ratio.append(np.nan); continue
            theta_max, _ = get_theta_max_beta_from_tbm(Ms[-1], gamma)
            if theta >= theta_max:
                Ms.append(np.nan); P_ratio.append(np.nan)
            else:
                _, P2_1, _, M2, _ = mach_function(Ms[-1], gamma, theta)
                Ms.append(M2); P_ratio.append(P2_1)
        elif theta < 0.0:
            if Ms[-1] < 1.0 or np.isnan(Ms[-1]):
                Ms.append(np.nan); P_ratio.append(np.nan)
            else:
                M2 = get_M2_from_nu(Ms[-1], gamma, -theta)
                P02_P2 = P0_P(M2, gamma)
                P01_P1 = P0_P(Ms[-1], gamma)
                Ms.append(M2); P_ratio.append(P01_P1 / P02_P2)
        else:
            Ms.append(Ms[-1]); P_ratio.append(1.0)

    return np.array(Ms), np.array(P_ratio)

def get_drag_so_far(x, y, top, M_in, gamma, depth, P_inf, T_inf):
    _rho_inf = P_inf / (R_air * T_inf)
    _a_inf   = get_speed_of_sound(T_inf, R_air, gamma)
    _V_inf   = M_in * _a_inf

    surface_angles = get_surface_angles(x, y, aoa=0)
    Ms, P_ratios = get_Mach_numbers_p_ratios(x, y, top, M_in, gamma)

    D = 0.0
    Pi_P0 = 1.0
    for i in range(len(surface_angles)):
        if i < len(P_ratios):
            Pi_P0 *= P_ratios[i]
        length = np.hypot(x[i+1]-x[i], y[i+1]-y[i])
        s = np.sin(np.deg2rad(surface_angles[i]))
        if top:
            D += length * depth * s * (Pi_P0 * P_inf)
        else:
            D -= length * depth * s * (Pi_P0 * P_inf)
    return D

def generate_shape(y_start, num_points, M_in, gamma, depth, P_inf, T_inf):
    x = np.linspace(0.0, 1.0, num_points)
    dx = x[1] - x[0]

    y_top = np.zeros_like(x)
    y_bot = np.zeros_like(x)
    y_top[0] = y_start
    y_bot[0] = y_start

    M_top_now = M_in
    M_bot_now = M_in

    for i in range(num_points - 1):
        # Guard: avoid invalid Machs
        if M_top_now <= 1.0 or np.isnan(M_top_now):
            theta_top_max_deg = 0.0
        else:
            theta_top_max_deg, _ = get_theta_max_beta_from_tbm(M_top_now, gamma)

        if M_bot_now <= 1.0 or np.isnan(M_bot_now):
            theta_bot_max_deg = 0.0
        else:
            theta_bot_max_deg, _ = get_theta_max_beta_from_tbm(M_bot_now, gamma)

        y_top_span = dx * np.tan(np.radians(theta_top_max_deg))
        y_bot_span = dx * np.tan(np.radians(theta_bot_max_deg))

        y_top_candidates = np.linspace(y_top[0]+tol, min(1.0, y_top[i] + y_top_span), 21)
        y_bot_candidates = np.linspace(max(0.0, y_bot[i] - y_bot_span), y_bot[0]-tol, 21)

        # evaluate top
        costs_top = []
        top_vals = []
        for ytn in y_top_candidates:
            y_top_trial = np.concatenate([y_top[:i], [ytn]])
            x_trial = x[:i+1]

            M_tops, _ = get_Mach_numbers_p_ratios(x_trial, y_top_trial, True, M_in, gamma)

            # Strict constraint: must have Mach > 1 at exit
            if M_tops[-1] <= 1.0 or np.isnan(M_tops[-1]):
                costs_top.append(np.inf)
                top_vals.append(ytn)
                continue

            try:
                D_top = get_drag_so_far(x_trial, y_top_trial, True, M_in, gamma, depth, P_inf, T_inf)
            except:
                D_top = np.nan

            area_top = np.trapz(y_top_trial, x_trial) - x[i+1] * (y_start)
            box_above = x[i+1] * (1 - y_start)
            N_above = 2 * (np.ceil(box_above / area_top)) - 1
            C_top = 20.0 * N_above * max(0.0, D_top)
            costs_top.append(C_top); top_vals.append(ytn)

        y_top[i+1] = top_vals[np.nanargmin(costs_top)]

        # evaluate bottom
        costs_bot = []
        bot_vals = []
        for ybn in y_bot_candidates:
            ybn_clip = min(ybn, y_top[i+1] - 1e-8)
            y_bot_trial = np.concatenate([y_bot[:i], [ybn_clip]])
            x_trial = x[:i+1]

            M_bots, _ = get_Mach_numbers_p_ratios(x_trial, y_bot_trial, False, M_in, gamma)

            # Strict constraint: must have Mach > 1 at exit
            if M_bots[-1] <= 1.0 or np.isnan(M_bots[-1]):
                costs_bot.append(np.inf)
                bot_vals.append(ybn_clip)
                continue

            try:
                D_bot = get_drag_so_far(x_trial, y_bot_trial, False, M_in, gamma, depth, P_inf, T_inf)
            except:
                D_bot = np.nan

            area_bot = x[i+1] * (y_start) - np.trapz(y_bot_trial, x_trial)
            box_below = x[i+1] * (y_start)
            N_below = 2 * (np.ceil(box_below / area_bot)) -1
            C_bot = 20.0 * N_below * max(0.0, D_bot)
            costs_bot.append(C_bot); bot_vals.append(ybn_clip)

        y_bot[i+1] = bot_vals[np.nanargmin(costs_bot)]

        # Update local Machs
        M_tops, _ = get_Mach_numbers_p_ratios(x[:i+1], y_top[:i+1], True, M_in, gamma)
        M_bots, _ = get_Mach_numbers_p_ratios(x[:i+1], y_bot[:i+1], False, M_in, gamma)
        M_top_now = M_tops[-1]
        M_bot_now = M_bots[-1]

    return x, y_top, y_bot

# Givens
M = 3
gamma = gamma_air
P = P_sea # Pa
T = 300 # K
L = 1 # m, depth into page

x, y_top, y_bot = generate_shape(0.5, 50, M, gamma, L, P, T)

print(x, y_top, y_bot)

N, D, cost = cost_function(x, y_top, y_bot, M, gamma, P, T, L, inviscid=True)

print(cost)

plt.figure()
plt.plot(x, y_top, label='Top Surface')
plt.plot(x, y_bot, label='Bottom Surface')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()