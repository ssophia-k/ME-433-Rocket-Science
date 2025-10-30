import numpy as np
import matplotlib.pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0, os.fspath(Path(__file__).parents[1]))

from Tools.oblique_shock import *
from Tools.expansion_fan import *
from Tools.isentropic import P0_P, T0_T, rho0_rho
from Tools.misc_functions import get_speed_of_sound
from HW4.P3_helpers import get_surface_angles, get_turn_angles
from Tools.constants import *
from Midterm.cost import cost_function

tol = 1e-6

def get_Mach_numbers_p_ratios(x, y, top, M_in, gamma, want_rho_temp=False):
    """
    Get the Mach number and P_n+1 / P_n at each turn on the airfoil surface
    Parameters:
        M_in: incoming Mach number (unitless)
        gamma: ratio of specific heats (unitless)
        top_thetas: turning angles at each turn on top surface (deg)
        bottom_thetas: turning angles at each turn on bottom surface (deg)
    Returns:
        M_top: Mach numbers at each state on top surface (array of n elements)
        P_ratio_top: pressure ratios at each state on top surface (array of n-1 elements)
        M_bottom: Mach numbers at each state on bottom surface (array of n elements)
        P_ratio_bottom: pressure ratios at each state on bottom surface (array of n-1 elements)

        If M = [M1, M2, M3], then P_ratio = [P2/P1, P3/P2]
    """
    Ms = [M_in]
    P_ratio = []
    thetas = get_turn_angles(x, y, aoa=0, top=top)
    rho_ratio = []
    T_ratio = []
    for theta in thetas:
        # if last Mach number is already NaN or <1, propagate NaNs forward
        if np.isnan(Ms[-1]) or Ms[-1]<1.0:
            Ms.append(np.nan)
            P_ratio.append(np.nan)
            continue

        if theta > (0 + tol):  # oblique shock
            theta_max, beta_max = get_theta_max_beta_from_tbm(Ms[-1], gamma)
            if theta >= theta_max:  # detached shock, no solution
                Ms.append(np.nan)
                P_ratio.append(np.nan)
                rho_ratio.append(np.nan)
                T_ratio.append(np.nan)
            else:
                #np.rad2deg(beta), P_out_P_in, T_out_T_in, M_out, rho_out_rho_in
                beta, P2_1, T2_1, M2, rho2_1 = mach_function(Ms[-1], gamma, theta) 
                Ms.append(M2)
                P_ratio.append(P2_1)
                rho_ratio.append(rho2_1)
                T_ratio.append(T2_1)

        elif theta < (0 - tol):  # expansion fan
            try:
                M2 = get_M2_from_nu(Ms[-1], gamma, -theta)  # -theta because theta is negative
            except:
                Ms.append(np.nan)
                P_ratio.append(np.nan)
                rho_ratio.append(np.nan)
                T_ratio.append(np.nan)
                continue
            Ms.append(M2)
            P02_P2 = P0_P(Ms[-1], gamma)
            P01_P1 = P0_P(Ms[-2], gamma)
            # isentropic expansion, P02 = P01. so P2/P1 = P01/P1 * 1/(P02/P2)
            P_ratio.append(P01_P1/P02_P2)  

            T02_T2 = T0_T(Ms[-1], gamma)
            T01_T1 = T0_T(Ms[-2], gamma)
            T_ratio.append(T01_T1/T02_T2) 

            rho02_rho2 = rho0_rho(Ms[-1], gamma)
            rho01_rho1 = rho0_rho(Ms[-2], gamma)
            rho_ratio.append(rho01_rho1/rho02_rho2) 
        else:  # theta == 0, no change in Mach number
            Ms.append(Ms[-1])
            P_ratio.append(1.0)
            T_ratio.append(1.0)
            rho_ratio.append(1.0)
    
    if want_rho_temp:
        return np.array(Ms), np.array(P_ratio), np.array(rho_ratio), np.array(T_ratio)
    else:
        return np.array(Ms), np.array(P_ratio)
    
def drag(x, y, top, M_in, gamma, P_inf, T_inf, depth, inviscid = True):
    surface_angles = get_surface_angles(x, y, aoa=0)
    turn_angles = get_turn_angles(x, y, aoa=0, top=top)

    """
      if want_rho_temp:
        return np.array(M_top), np.array(P_ratio_top), np.array(rho_ratio_top), np.array(T_ratio_top), np.array(M_bot), np.array(P_ratio_bot), np.array(rho_ratio_bot), np.array(T_ratio_bot)
    """
    Ms, P_ratio, rho_ratio, T_ratio = get_Mach_numbers_p_ratios(x, y, top, M_in, gamma, want_rho_temp=True)
    if (np.isnan(Ms).any() or np.isnan(P_ratio).any()):
        return np.nan

    D=0
    Pi_P0 = 1.0  # P1 / P_inf
    rhoi_rho0 = 1.0
    Ti_T0 = 1.0

    for i in range(len(surface_angles)): # length of 1 less than x,y arrays (I think)
        
        Pi_P0 *= P_ratio[i] 

        rho_inf = P_inf / (R_air * T_inf)
        rhoi_rho0 *= rho_ratio[i]
        rho = rhoi_rho0 * rho_inf

        Ti_T0   *= T_ratio[i]

        T = Ti_T0  * T_inf

        a = get_speed_of_sound(T, R_air, gamma)

        V = Ms[i+1] * a


        length = np.sqrt( (x[i+1]-x[i])**2 + (y[i+1]-y[i])**2 ) #pythag


        if top:
            # For top surface, sin will cover our +- surface angles appropriately
            D += length * depth * np.sin(np.deg2rad(surface_angles[i])) * Pi_P0 * P_inf
        else: 
            # For bottom surface, just need to negate the sin term
            D -= length * depth * np.sin(np.deg2rad(surface_angles[i])) * Pi_P0 * P_inf

        if inviscid == False:
            D += 0.5 * depth * length * rho * V**2 * 0.01

    return D

def generate_shape(y_start, num_points, M_in, gamma, depth, P_inf, T_inf, inviscid=True):
    x = np.linspace(0.0, 1.0, num_points)
    dx = x[1] - x[0]

    y_top = np.zeros_like(x)
    y_bot = np.zeros_like(x)
    y_top[0] = y_start
    y_bot[0] = y_start

    M_tops = [M_in]
    M_bots = [M_in]

    for i in range(num_points - 1):
        M_top_now = M_tops[-1]
        M_bot_now = M_bots[-1]
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

        y_top_candidates = np.linspace(y_top[0]+tol, min(1.0, y_top[i] + y_top_span-tol), 21)
        y_bot_candidates = np.linspace(max(0.0, y_bot[i] - y_bot_span)+tol, y_bot[0]-tol, 21)

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
                D_top = drag(x_trial, y_top_trial, True, M_in, gamma, depth, P_inf, T_inf, inviscid)
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
                D_bot = drag(x_trial, y_bot_trial, False, M_in, gamma, depth, P_inf, T_inf, inviscid)
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

    return x, y_top, y_bot

# Givens
M = 3
gamma = gamma_air
P = P_sea # Pa
T = 300 # K
L = 1 # m, depth into page

x, y_top, y_bot = generate_shape(0.5, 10, M, gamma, L, P, T)

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