import numpy as np

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.numerical_iterator import *

def get_theta_max_beta_from_tbm(M_in, gamma):
    """
    Get the beta at which maximum deflection angle (theta_max) occurs
    Parameters:
        M_in: incoming Mach number (unitless)
        gamma: ratio of specific heats (unitless)
    Returns:
        theta_max_deg: maximum deflection angle (deg)
    """
    if M_in <= 1:
        raise ValueError("M_in must be > 1")

    beta = np.linspace(np.arcsin(1.0/M_in) + 1e-6, np.pi/2 - 1e-6, 20000)

    sinb = np.sin(beta)
    cosb = np.cos(beta)
    cotb = cosb / sinb
    cos2b = np.cos(2*beta)

    num = M_in**2 * sinb**2 - 1.0
    den = M_in**2 * (gamma + cos2b) + 2.0
    tan_theta = 2.0 * cotb * (num / den)

    # mask invalid regions
    tan_theta[~np.isfinite(tan_theta)] = np.nan
    theta = np.arctan(tan_theta)

    # keep only positive turning angles
    theta[theta <= 0] = np.nan

    if np.all(np.isnan(theta)):
        raise RuntimeError(f"No valid theta for M={M_in}, gamma={gamma}")

    idx = np.nanargmax(theta)
    return np.degrees(theta[idx]), np.degrees(beta[idx])


# Not necessary anymore after I found analytical sln in textbook
# But keeping here just in case
# def get_weak_beta_from_tbm(M_in, gamma, theta_deg):
#     """
#     Solve the theta-beta-Mach relation for beta. Weak solution only.
#     Parameters:
#         M: incoming Mach number (unitless)
#         theta_deg: turning angle (deg)
#         gamma: ratio of specific heats (unitless)
#     Returns:
#         beta_deg: shock angle (deg)
#     """
#     theta_max_deg, beta_max_deg = get_theta_max_beta_from_tbm(M_in, gamma)

#     # Preconditions
#     if M_in <= 1:
#         raise ValueError("M_in must be > 1 for oblique shocks.")
#     if theta_deg > theta_max_deg:
#         raise ValueError(f"theta_deg must be <= theta_max ({theta_max_deg:.2f} deg) for M_in={M_in}, gamma={gamma}.")
#     theta = np.radians(theta_deg)
#     if theta <= 0:
#         return np.degrees(np.arcsin(1.0 / M_in))  # Mach angle

#     def func(beta):
#         left = np.tan(theta)
#         right = 2 * (1/np.tan(beta)) * ( (M_in**2 * np.sin(beta)**2 - 1) / 
#                                         (M_in**2 * (gamma + np.cos(2*beta))+2))
#         return left - right # want this to be zero in order to solve numerically

#     # in order to only return the weak solution, we must limit the search
#     # beta starts at sonic boom angle which is sin^-1(1/M)
#     beta_sonic_boom = np.arcsin(1/M_in) + 1e-6 
#     # check beta between 0 and beta at theta_max beta value
#     # this way we only return the weak solution
#     solution = numerical_iterator(func=func, start=beta_sonic_boom, end =np.deg2rad(beta_max_deg), goal_y=0, tol=1e-6, max_iter=10000)
#     beta_deg = np.degrees(solution)
#     return beta_deg
    


def mach_function(M_in, gamma, theta_deg, delta = 1):
    """
    Assumptions: Calorically perfect ideal gas. Steady, adiabatic, inviscid flow.
    Parameters:
        M_in: incoming Mach number (unitless)
        gamma: ratio of specific heats (unitless)
        theta_deg: turning angle in degrees, measured from the line of the incoming flow
                Positive theta turns the flow towards itself (concave corner)
                Negative theta turns the flow away from itself (convex corner)
                This is also the angle between the incoming and outgoing flow directions
                (since outflow will follow the wall boundary)
        delta: weak (1) or strong (0) shock solution
    Returns:
        beta: shock angle (deg)
        P_out_P_in: outgoing to incoming pressure ratio (unitless)
        T_out_T_in: outgoing to incoming temperature ratio (unitless)
        M_out: outgoing Mach number (unitless)
    """

    if delta not in [0, 1]:
        raise ValueError("delta must be 0 (strong shock) or 1 (weak shock)")
    if theta_deg < 0:
        raise ValueError("theta_deg must be >= 0 for oblique shocks")
    if M_in <= 1:
        raise ValueError("M_in must be > 1 for oblique shocks")
    
    theta = np.deg2rad(theta_deg)

    if theta == 0: #return normal shock solution
        beta = np.pi/2
        rho_out_rho_in = ( (gamma+1) * M_in**2) / ( 2 + (gamma-1) * M_in**2)
        P_out_P_in = 1 + ( (2 * gamma) / (gamma + 1)) * (M_in**2 - 1)
        T_out_T_in = P_out_P_in / rho_out_rho_in
        M_out = np.sqrt( (1 + (gamma-1)/2 * M_in**2) / (gamma * M_in**2 - (gamma-1)/2))
    else:
        lamb = ( (M_in**2 -1)**2 - 3*(1+ (gamma-1)/2 * M_in**2) * (1 + (gamma+1)/2 * M_in**2) * np.tan(theta)**2 ) ** (1/2)
        chi = (1/lamb**3) * ( (M_in**2 -1)**3 - 9*(1+ (gamma-1)/2 * M_in**2) * (1+ (gamma-1)/2 * M_in**2 + (gamma+1)/4 * M_in**4) * np.tan(theta)**2)
        tan_beta = (1/ (3* (1+ (gamma-1)/2 * M_in**2) * np.tan(theta))) * ( M_in**2 -1 + 2*lamb * np.cos( (4*np.pi*delta + np.arccos(chi))/3))
        beta = np.arctan(tan_beta)

        M_in_n = M_in * np.sin(beta) # normal component of incoming Mach number

        # Shock relations
        rho_out_rho_in = ( (gamma + 1) * M_in_n**2) / ( (gamma-1) * M_in_n**2 +2)
        P_out_P_in = 1 + ( (2 * gamma) / (gamma + 1)) * (M_in_n**2 - 1)
        M_out_n = np.sqrt( (M_in_n**2 + (2/ (gamma-1))) / ( (2*gamma/(gamma-1)) * M_in_n**2 - 1)) 
        T_out_T_in = P_out_P_in / rho_out_rho_in
        M_out = M_out_n / np.sin(beta - theta)
    
    return np.rad2deg(beta), P_out_P_in, T_out_T_in, M_out

    

    