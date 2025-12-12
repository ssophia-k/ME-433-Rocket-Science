# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 13:51:30 2025

@author: Adin Sacho-Tanzer
"""

from Tools.misc_functions import get_speed_of_sound
from Tools.constants import R_air
from Tools.oblique_shock import mach_function

import numpy as np

def calculate_thrust(inlet, P_in, M_in, T_in, P_out, M_out, T_out, A_out, front_length, front_angle, width):
    """
    Calculate thrust
    Parameters
    inlet : inlet object
    P_in : incoming atmospheric pressure, Pa
    M_in : incoming atmospheric mach number
    T_in : incoming atmospheric temperature, K
    P_out : pressure at back of nozzle, Pa
    M_out : mach number at back of nozzle
    T_out : temperature at back of nozzle
    A_out : area of back of nozzle
    front_length : area of front panel
    front_angle : angle of front panel
    width : width of engine
    Returns
    thrust : thrust, Ns
    """
    # thrust_estimate = (P6*A6s[-1] + m_dot*get_speed_of_sound(T6)*M6) - (P_atm*(inlet.y_lip-0)*width+m_dot*get_speed_of_sound(T_atm)*M_atm)
    # print(f"Thrust estimate: {thrust_estimate} N")
    pressure_force_inlet = inlet.get_pressure_drag(P_in, T_in, M_in)
    momentum_flux_inlet = inlet.get_inlet_momentum_flux(P_in, T_in, M_in)
    
    pressure_force_outlet = P_out*A_out  # Note: this doesn't account for any possible wall thickness at the outlet
    rho_outlet = P_out/(R_air*T_out)
    a_outlet = get_speed_of_sound(T_out)
    momentum_flux_outlet = rho_outlet*(a_outlet*M_out)**2*A_out
    
    _, Pr, _, _, _ = mach_function(M_in, 1.4, front_angle)
    P_bottom = Pr*P_in
    pressure_force_bottom = P_bottom*front_length*width*np.sin(np.radians(front_angle))
    
    thrust = momentum_flux_outlet + pressure_force_outlet - momentum_flux_inlet - pressure_force_inlet - pressure_force_bottom
    # print(f"Actual thrust: {thrust} N")
    return thrust
