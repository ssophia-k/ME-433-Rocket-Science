import numpy as np
import matplotlib.pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))
from Tools.oblique_shock import *
from Tools.expansion_fan import *
from Tools.isentropic import P0_P
from Tools.misc_functions import get_speed_of_sound
from Tools.airfoil_eval import *
from Tools.constants import *

def drag(x, y_top, y_bot, M_in, gamma, P_inf, T_inf, depth, inviscid = True):
    surface_angles_top = get_surface_angles(x, y_top, aoa=0)
    surface_angles_bot = get_surface_angles(x,y_bot,aoa=0)
    turn_angles_top = get_turn_angles(x,y_top,aoa=0, top=True)
    turn_angles_bot = get_turn_angles(x,y_bot,aoa=0,top=False)

    """
      if want_rho_temp:
        return np.array(M_top), np.array(P_ratio_top), np.array(rho_ratio_top), np.array(T_ratio_top), np.array(M_bot), np.array(P_ratio_bot), np.array(rho_ratio_bot), np.array(T_ratio_bot)
    """
    M_top, P_ratio_top, rho_ratio_top, T_ratio_top, M_bot, P_ratio_bot, rho_ratio_bot, T_ratio_bot = get_Mach_numbers_p_ratios(M_in, gamma, turn_angles_top, turn_angles_bot, want_rho_temp=True)
    if (np.isnan(M_top).any() or np.isnan(P_ratio_top).any() or np.isnan(M_bot).any() or np.isnan(P_ratio_bot).any()):
        return np.nan

    D=0
    Pi_P0_top = 1.0  # P1 / P_inf
    Pi_P0_bot = 1.0
    rhoi_rho0_top = 1.0
    rhoi_rho0_bot = 1.0
    Ti_T0_top = 1.0
    Ti_T0_bot = 1.0
    for i in range(len(surface_angles_top)): # length of 1 less than x,y arrays (I think)
        
        Pi_P0_top *= P_ratio_top[i] 
        Pi_P0_bot *= P_ratio_bot[i]

        rho_inf = P_inf / (R_air * T_inf)
        rhoi_rho0_top *= rho_ratio_top[i]
        rhoi_rho0_bot *= rho_ratio_bot[i]
        rho_top = rhoi_rho0_top * rho_inf
        rho_bot = rhoi_rho0_bot * rho_inf


        Ti_T0_top   *= T_ratio_top[i]
        Ti_T0_bot   *= T_ratio_bot[i]
        T_top = Ti_T0_top * T_inf
        T_bot = Ti_T0_bot * T_inf
        a_top = get_speed_of_sound(T_top, R_air, gamma)
        a_bot = get_speed_of_sound(T_bot, R_air, gamma)
        V_top = M_top[i+1] * a_top
        V_bot = M_bot[i+1] * a_bot

        length_top = np.sqrt( (x[i+1]-x[i])**2 + (y_top[i+1]-y_top[i])**2 ) #pythag
        length_bot = np.sqrt( (x[i+1]-x[i])**2 + (y_bot[i+1]-y_bot[i])**2 )

        # For top surface, sin will cover our +- surface angles appropriately
        D += length_top * depth * np.sin(np.deg2rad(surface_angles_top[i])) * Pi_P0_top * P_inf
        # For bottom surface, just need to negate the sin term
        D -= length_bot * depth * np.sin(np.deg2rad(surface_angles_bot[i])) * Pi_P0_bot * P_inf

        if inviscid == False:
            D += 0.5 * depth * length_top * rho_top * V_top**2 * 0.01 * np.cos(np.deg2rad(surface_angles_top[i]))
            D += 0.5 * depth * length_bot * rho_bot * V_bot**2 * 0.01 * np.cos(np.deg2rad(surface_angles_bot[i]))
    return D

def enclosed_area(x, y_top, y_bot):
    top_area = np.trapz(y_top, x)
    bottom_area = np.trapz(y_bot,x)
    # height = np.array(y_top) - np.array(y_bot)
    # return np.trapz(height, x)
    return top_area-bottom_area

def cost_function(x, y_top, y_bot, M_in, gamma, P_inf, T_inf, depth, inviscid=True):
    """
    Calculate the cost function for a given airfoil shape. 
    Needs top and bottom surfaces.
    """
    D = drag(x, y_top, y_bot, M_in, gamma, P_inf, T_inf, depth, inviscid)
    if np.isnan(D):
        return np.inf, np.inf, np.inf
    A = enclosed_area(x, y_top, y_bot)
    if A <= 0.0:
        return np.inf, np.inf, np.inf
    
    N = 2 * np.ceil(1/A) -1 # number of trips to move entire volume. depth cancels out. big area is 1m^3

    return N, D, (20.0 * N * D)

    


