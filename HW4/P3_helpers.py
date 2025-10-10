import numpy as np
import matplotlib.pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.oblique_shock import *
from Tools.expansion_fan import *
from Tools.more_functions import P0_P


def get_surface_angles(x, y, aoa, top=True):
    """
    Get the local surface angle (theta) at each turn on the airfoil
    Parameters:
        x: x-coordinates of airfoil surface (array)
        y: y-coordinates of airfoil surface (array)
        aoa: angle of attack (deg)
        top: True if top surface, False if bottom surface (bool)
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
    
    return surface_theta if top else -surface_theta


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
    theta = get_surface_angles(x,y,aoa,top)
    turn_angles = []
    for i in range(len(theta)):
        if i == 0:
            # first flow comes in at horizontal, so the first turn is just the surface_theta
            turn_angles.append(theta[i])
        else:
             # both angles positive wrt the horizontal
            if (theta[i] >= 0) & (theta[i-1] >= 0):
                if (theta[i]>theta[i-1]): # we are becoming more steep. aka oblique shock.
                    turn_angles.append(theta[i]-theta[i-1]) # theta2-theta1
                elif (theta[i]<theta[i-1]): # we are becoming less steep. but both above horizontal
                    angle = theta[i-1]-theta[i] # theta1-theta2. positive
                    turn_angles.append(-angle) # sign it negative to denote expansion wave

            # angle 1 above horizontal, angle 2 below horizontal
            elif (theta[i-1]>0) & (theta[i]<0): 
                angle = theta[i-1] - theta[i] # total angle between the two is adding, but second one is neg
                turn_angles.append(-angle) #sign it negative to denote expansion wave

            # angle 1 below horizontal, angle 2 above horizontal (this never happens but still)
            elif (theta[i-1]<0) & (theta[i]>0):
                turn_angles.append(180 - theta[i] + theta[i-1]) # oblique shock

            #both are below the horizontal
            else: 
                if (theta[i]>theta[i-1]): # both negative, angle 2 is steeper (more negative)
                    angle = np.abs(theta[i])-np.abs(theta[i-1])
                    turn_angles.append(-angle) # expansion wave
                elif (theta[i]<theta[i-1]): # both negative, angle 2 is less steep and less negative
                    angle = np.abs(theta[i-1])-np.abs(theta[i])
                    turn_angles.append(angle) # positive, oblique shock

    return turn_angles


def get_Mach_numbers_p_ratios(M_in, gamma, top_thetas, bottom_thetas):
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
    for theta in top_thetas:
        # if last Mach number is already NaN, propagate NaNs forward
        if np.isnan(M_top[-1]):
            M_top.append(np.nan)
            P_ratio_top.append(np.nan)
            continue

        if theta > 0:  # oblique shock
            theta_max, beta_max = get_theta_max_beta_from_tbm(M_top[-1], gamma)
            if theta >= theta_max:  # detached shock, no solution
                M_top.append(np.nan)
                P_ratio_top.append(np.nan)
            else:
                beta, P2_1, T2_1, M2 = mach_function(M_top[-1], gamma, theta)
                M_top.append(M2)
                P_ratio_top.append(P2_1)
        elif theta < 0:  # expansion fan
            M2 = get_M2_from_nu(M_top[-1], gamma, -theta)  # -theta because theta is negative
            M_top.append(M2)
            P02_P2 = P0_P(M_top[-1], gamma)
            P01_P1 = P0_P(M_top[-2], gamma)
            # isentropic expansion, P02 = P01. so P2/P1 = P01/P1 * 1/(P02/P2)
            P_ratio_top.append(P01_P1/P02_P2)  
        else:  # theta == 0, no change in Mach number
            M_top.append(M_top[-1])
            P_ratio_top.append(1.0)
    
    M_bottom = [M_in]
    P_ratio_bottom = []
    for theta in bottom_thetas:
        # if last Mach number is already NaN, propagate NaNs forward
        if np.isnan(M_bottom[-1]):
            M_bottom.append(np.nan)
            P_ratio_bottom.append(np.nan)
            continue

        if theta > 0:  # oblique shock
            theta_max, beta_max = get_theta_max_beta_from_tbm(M_bottom[-1], gamma)
            if theta >= theta_max:  # detached shock, no solution
                M_bottom.append(np.nan)
                P_ratio_bottom.append(np.nan)
            else:
                beta, P2_1, T2_1, M2 = mach_function(M_bottom[-1], gamma, theta)
                M_bottom.append(M2)
                P_ratio_bottom.append(P2_1)
        elif theta < 0:  # expansion fan
            M2 = get_M2_from_nu(M_bottom[-1], gamma, -theta)  # -theta because theta is negative
            M_bottom.append(M2)
            P02_P2 = P0_P(M_bottom[-1], gamma)
            P01_P1 = P0_P(M_bottom[-2], gamma)
            # isentropic expansion, P02 = P01. so P2/P1 = P01/P1 * 1/(P02/P2)
            P_ratio_bottom.append(P01_P1/P02_P2)  
        else:  # theta == 0, no change in Mach number
            M_bottom.append(M_bottom[-1])
            P_ratio_bottom.append(1.0)
    
    return np.array(M_top), np.array(P_ratio_top), np.array(M_bottom), np.array(P_ratio_bottom)



def get_cL_and_cD(M_in, gamma, x, y, aoa, P_inf=101325.0):
    surface_angles_top = get_surface_angles(x, y, aoa, True)
    surface_angles_bot = get_surface_angles(x,-y,aoa,False)
    turn_angles_top = get_turn_angles(x,y,aoa,True)
    turn_angles_bot = get_turn_angles(x,-y,aoa,False)

    M_top, P_ratio_top, M_bot, P_ratio_bot = get_Mach_numbers_p_ratios(M_in, gamma, turn_angles_top, turn_angles_bot)
    
    # These are all still normalized by P1
    # which is fine since it is otherwise linear addition of pressures with coeff
    # So this is lift/drag per unit into the page per P1
    C_L = []
    C_D = []
    for i in range(len(surface_angles_top)): # length of 1 less than x,y arrays (I think)
        L=0
        D=0

        length = np.sqrt( (x[i+1]-x[i])**2 + (y[i+1]-y[i])**2 ) #pythag

        # all top pressures will contribute negative lift
        L -= length * np.cos(np.deg2rad(np.abs(surface_angles_top[i]))) * P_ratio_top[i]
        # all bottom pressures contribute pos lift
        L += length * np.cos(np.deg2rad(np.abs(surface_angles_bot[i]))) * P_ratio_bot[i]
    
        # Now look at drag
        if surface_angles_top[i] >=0 : 
            D += length * np.sin(np.deg2rad(np.abs(surface_angles_top[i]))) * P_ratio_top[i]
        else: 
            D -= length * np.sin(np.deg2rad(np.abs(surface_angles_top[i]))) * P_ratio_top[i]

        # I made it so a positive surface angle on the bottom is pointing below horizontal.
        # so still adding to drag
        if surface_angles_bot[i] >=0 : 
            D += length * np.sin(np.deg2rad(np.abs(surface_angles_bot[i]))) * P_ratio_bot[i]
        else: 
            D -= length * np.sin(np.deg2rad(np.abs(surface_angles_bot[i]))) * P_ratio_bot[i]

     # geometry and dynamic pressure
    x = np.asarray(x); y = np.asarray(y)
    chord = float(np.max(x) - np.min(x))
    A = chord * 1.0  # 1 m span into page
    q_inf = 0.5 * gamma * (M_in**2) * P_inf
    
    # coefficients
    cL = L / (q_inf * A)
    cD = D / (q_inf * A)
    return cL, cD



        




"""
UNUSED OOPS
"""
def get_stag_pressure_ratios(M, gamma):
    """
    Get the pressure ratio at each state on the airfoil surface
    Parameters:
        M: Mach numbers at each state on surface (array)
        gamma: ratio of specific heats (unitless)
    Returns:
        P0_P_arr: stagnation pressure ratios at each state on surface (array)
    """
    P0_P_arr = []
    for M_i in M:
            P0_P_arr.append(P0_P(M_i, gamma))
    return np.array(P0_P_arr)


def get_end_pressure_ratios(x, y, gamma, M_top, P_ratio_top, M_bottom, P_ratio_bottom):
    P_n_ratio_top = np.prod(P_ratio_top)  # P_last_turn / P_in for top surface
    P_n_ratio_bottom = np.prod(P_ratio_bottom)  # P_last_turn / P_in for bottom surface

    # Now need to get P_n+1 / P_in. here we need to make sure that they converge at the trailing edge
    # theta at inner trailing edge is 2arctan(dydx) at the end since symmmetrical airgoil
    inner_theta_end = 2*np.degrees(np.arctan((y[-2]-y[-1]) / (x[-2]-x[-1])))

    # Then, theta_top = turning angle for top stream from centerline
    # theta_bot = turning angle for bottom stream from centerline, = inner_theta_end - theta_top
    def f_n_to_end_top(theta_top): #returns P_end/Pin
        P_end_in = P_n_ratio_top
        if theta_top > 0:  # oblique shock
            theta_max, beta_max = get_theta_max_beta_from_tbm(M_top[-1], gamma)
            if theta_top >= theta_max:  # detached shock, no solution
                return np.nan
            else:
                beta, P2_1, T2_1, M2 = mach_function(M_top[-1], gamma, theta_top)
                P_end_in *= P2_1
        elif theta_top < 0:  # expansion fan
            M2 = get_M2_from_nu(M_top[-1], gamma, -theta_top)  # -theta because theta is negative
            P02_P2 = P0_P(M2, gamma)
            P01_P1 = P0_P(M_top[-1], gamma)
            P_end_in *= P01_P1/P02_P2 # isentropic expansion, P02 = P01. so P2/P1 = P01/P1 * 1/ (P02/P2)
        else:  # theta == 0, no change in Mach number
            pass
        return P_end_in
        
        
    def f_n_to_end_bottom(theta_top):
        theta_bottom = inner_theta_end-theta_top
        P_end_in = P_n_ratio_bottom
        if theta_bottom > 0:  # oblique shock
            theta_max, beta_max = get_theta_max_beta_from_tbm(M_bottom[-1], gamma)
            if theta_bottom >= theta_max:  # detached shock, no solution
                return np.nan
            else:
                beta, P2_1, T2_1, M2 = mach_function(M_bottom[-1], gamma, theta_bottom)
                P_end_in *= P2_1
        elif theta_bottom < 0:  # expansion fan
            M2 = get_M2_from_nu(M_bottom[-1], gamma, -theta_bottom)  # -theta because theta is negative
            P02_P2 = P0_P(M2, gamma)
            P01_P1 = P0_P(M_bottom[-1], gamma)
            P_end_in *= P01_P1/P02_P2 # isentropic expansion, P02 = P01. so P2/P1 = P01/P1 * 1/ (P02/P2)
        else:  # theta == 0, no change in Mach number
            pass
        return P_end_in
    
    # Need to iterate values of theta_top / theta_bottom until we converge P_end_in_top = P_end_in_bottom
    def func(theta_top):
        return f_n_to_end_top(theta_top) - f_n_to_end_bottom(theta_top)
    
    # Use numerical iterator to find root
    # def numerical_iterator(func, start, end, goal_y, tol=1e-6, max_iter=1000):
    theta_top = numerical_iterator(func, -20, 20, 0, tol=1e-6, max_iter=1000)

    return theta_top, f_n_to_end_top(theta_top)