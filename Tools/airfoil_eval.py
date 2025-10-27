import numpy as np
import matplotlib.pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.oblique_shock import *
from Tools.expansion_fan import *
from Tools.isentropic import P0_P, T0_T, rho0_rho
from Tools.misc_functions import get_speed_of_sound
from Tools.constants import *

tol = 1e-8


def get_surface_angles(x, y, aoa):
    """
    Get the local surface angle at each turn on the airfoil
    Parameters:
        x: x-coordinates of airfoil surface (array)
        y: y-coordinates of airfoil surface (array)
        aoa: angle of attack (deg)
    Returns:
        surface_theta: local surface angle at each turn (deg), array
    """
    dy_dx = []
    for i in range(len(x)-1):
        dy_dx.append((y[i+1]-y[i])/(x[i+1]-x[i]))
    theta_wrt_centerline = np.degrees(np.arctan(dy_dx))  # angles wrt shape centerline

    # Adjust theta for angle of attack
    surface_theta = theta_wrt_centerline-aoa 
    # So each of these angles is with respect to the horizontal

    return surface_theta


def get_turn_angles(x, y, aoa, top=True):
    """
    Get the turn angles (so surface angle relative to previous angle of flow) on airfoil
    Parameters:
        x: x-coordinates of airfoil surface (array)
        y: y-coordinates of airfoil surface (array)
        aoa: angle of attack (deg)
        top: True if top surface, False if bottom surface (bool)
    Returns:
        theta: turn angle of flow (deg), array
    """
    theta = get_surface_angles(x,y,aoa)
    turn_angles = []
    for i in range(len(theta)):
        if i == 0:
            # first flow comes in at horizontal, so the first turn is just the surface_theta
            turn_angles.append(theta[i])
        else:
            turn_angles.append(theta[i]-theta[i-1])
    # Positive turn angle is O.S
    # Negative turn angle is E.W
    # Turn angle of 0 will be no change

    # Bottom surface will be negative of top
    if not top:
        turn_angles = [-t for t in turn_angles]

    return turn_angles


def get_Mach_numbers_p_ratios(M_in, gamma, top_thetas, bottom_thetas, want_rho_temp=False):
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
    M_top = [M_in]
    P_ratio_top = []
    rho_ratio_top = []
    T_ratio_top = []
    for theta in top_thetas:
        # if last Mach number is already NaN or <1, propagate NaNs forward
        if np.isnan(M_top[-1]) or M_top[-1]<1.0:
            M_top.append(np.nan)
            P_ratio_top.append(np.nan)
            continue

        if theta > (0 + tol):  # oblique shock
            theta_max, beta_max = get_theta_max_beta_from_tbm(M_top[-1], gamma)
            if theta >= theta_max:  # detached shock, no solution
                M_top.append(np.nan)
                P_ratio_top.append(np.nan)
                rho_ratio_top.append(np.nan)
                T_ratio_top.append(np.nan)
            else:
                #np.rad2deg(beta), P_out_P_in, T_out_T_in, M_out, rho_out_rho_in
                beta, P2_1, T2_1, M2, rho2_1 = mach_function(M_top[-1], gamma, theta) 
                M_top.append(M2)
                P_ratio_top.append(P2_1)
                rho_ratio_top.append(rho2_1)
                T_ratio_top.append(T2_1)

        elif theta < (0 - tol):  # expansion fan
            try:
                M2 = get_M2_from_nu(M_top[-1], gamma, -theta)  # -theta because theta is negative
            except:
                M_top.append(np.nan)
                P_ratio_top.append(np.nan)
                rho_ratio_top.append(np.nan)
                T_ratio_top.append(np.nan)
                continue
            M_top.append(M2)
            P02_P2 = P0_P(M_top[-1], gamma)
            P01_P1 = P0_P(M_top[-2], gamma)
            # isentropic expansion, P02 = P01. so P2/P1 = P01/P1 * 1/(P02/P2)
            P_ratio_top.append(P01_P1/P02_P2)  

            T02_T2 = T0_T(M_top[-1], gamma)
            T01_T1 = T0_T(M_top[-2], gamma)
            T_ratio_top.append(T01_T1/T02_T2) 

            rho02_rho2 = rho0_rho(M_top[-1], gamma)
            rho01_rho1 = rho0_rho(M_top[-2], gamma)
            rho_ratio_top.append(rho01_rho1/rho02_rho2) 
        else:  # theta == 0, no change in Mach number
            M_top.append(M_top[-1])
            P_ratio_top.append(1.0)
            T_ratio_top.append(1.0)
            rho_ratio_top.append(1.0)
    
    M_bot = [M_in]
    P_ratio_bot = []
    rho_ratio_bot = []
    T_ratio_bot = []
    for theta in bottom_thetas:
        # if last Mach number is already NaN or <1, propagate NaNs forward
        if np.isnan(M_bot[-1]) or M_bot[-1]<1.0:
            M_bot.append(np.nan)
            P_ratio_bot.append(np.nan)
            continue

        if theta > (0 + tol):  # oblique shock
            theta_max, beta_max = get_theta_max_beta_from_tbm(M_bot[-1], gamma)
            if theta >= theta_max:  # detached shock, no solution
                M_bot.append(np.nan)
                P_ratio_bot.append(np.nan)
                rho_ratio_bot.append(np.nan)
                T_ratio_bot.append(np.nan)
            else:
                #np.rad2deg(beta), P_out_P_in, T_out_T_in, M_out, rho_out_rho_in
                beta, P2_1, T2_1, M2, rho2_1 = mach_function(M_bot[-1], gamma, theta) 
                M_bot.append(M2)
                P_ratio_bot.append(P2_1)
                rho_ratio_bot.append(rho2_1)
                T_ratio_bot.append(T2_1)

        elif theta < (0 - tol):  # expansion fan
            try:
                M2 = get_M2_from_nu(M_bot[-1], gamma, -theta)  # -theta because theta is negative
            except:
                M_bot.append(np.nan)
                P_ratio_bot.append(np.nan)
                rho_ratio_bot.append(np.nan)
                T_ratio_bot.append(np.nan)
                continue
            M_bot.append(M2)
            P02_P2 = P0_P(M_bot[-1], gamma)
            P01_P1 = P0_P(M_bot[-2], gamma)
            # isentropic expansion, P02 = P01. so P2/P1 = P01/P1 * 1/(P02/P2)
            P_ratio_bot.append(P01_P1/P02_P2)  

            T02_T2 = T0_T(M_bot[-1], gamma)
            T01_T1 = T0_T(M_bot[-2], gamma)
            T_ratio_bot.append(T01_T1/T02_T2) 

            rho02_rho2 = rho0_rho(M_bot[-1], gamma)
            rho01_rho1 = rho0_rho(M_bot[-2], gamma)
            rho_ratio_bot.append(rho01_rho1/rho02_rho2) 
        else:  # theta == 0, no change in Mach number
            M_bot.append(M_bot[-1])
            P_ratio_bot.append(1.0)
            T_ratio_bot.append(1.0)
            rho_ratio_bot.append(1.0)
    
    if want_rho_temp:
        return np.array(M_top), np.array(P_ratio_top), np.array(rho_ratio_top), np.array(T_ratio_top), np.array(M_bot), np.array(P_ratio_bot), np.array(rho_ratio_bot), np.array(T_ratio_bot)
    else:
        return np.array(M_top), np.array(P_ratio_top), np.array(M_bot), np.array(P_ratio_bot)



def get_cL_and_cD(M_in, gamma, x, y, aoa, depth = 1, P_inf=P_sea, T_inf=300.0):
    rho_inf = P_inf / (R_air * T_inf)
    a_inf = get_speed_of_sound(T_inf, R_air, gamma)
    V_inf = M_in * a_inf


    surface_angles_top = get_surface_angles(x, y, aoa)
    surface_angles_bot = get_surface_angles(x,-y,aoa)
    turn_angles_top = get_turn_angles(x,y,aoa,True)
    turn_angles_bot = get_turn_angles(x,-y,aoa,False)

    M_top, P_ratio_top, M_bot, P_ratio_bot = get_Mach_numbers_p_ratios(M_in, gamma, turn_angles_top, turn_angles_bot)
    
    L=0
    D=0
    Pi_P0_top = 1.0  # P1 / P_inf
    Pi_P0_bot = 1.0
    tot_length = 0.0
    for i in range(len(surface_angles_top)): # length of 1 less than x,y arrays (I think)
        
        Pi_P0_top *= P_ratio_top[i] 
        Pi_P0_bot *= P_ratio_bot[i]
        length = np.sqrt( (x[i+1]-x[i])**2 + (y[i+1]-y[i])**2 ) #pythag
        tot_length += length

        # all top pressures will contribute negative lift
        L -= length * depth * np.cos(np.deg2rad(surface_angles_top[i])) * Pi_P0_top * P_inf
        # all bottom pressures contribute pos lift
        L += length * depth * np.cos(np.deg2rad(surface_angles_bot[i])) * Pi_P0_bot * P_inf
    
        # Now look at drag
        # For top surface, sin will cover our +- surface angles appropriately
        D += length * depth * np.sin(np.deg2rad(surface_angles_top[i])) * Pi_P0_top * P_inf
        # For bottom surface, just need to negate the sin term
        D -= length * depth * np.sin(np.deg2rad(surface_angles_bot[i])) * Pi_P0_bot * P_inf
        
    # Geometry
    A = tot_length * depth
    
    # coefficients
    cL = (2*L) / (rho_inf * V_inf**2 * A)
    cD = (2*D) / (rho_inf * V_inf**2 * A)
    return cL, cD
