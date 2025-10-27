import numpy as np

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.constants import *

# FOR ALL BOUNDARY STUFF, LET THE UPPER BOUND BELONG TO THE LOWER LAYER

def get_lapse_rate(altitude):
    """
    Returns the temperature gradient (lapse rate) at a given altitude.
    altitude [km]
    """
    if altitude <0:
        raise ValueError("Altitude cannot be negative.")
    elif altitude <= 11:
        lapse_rate = k_0_11
    elif altitude <= 20:
        lapse_rate = k_11_20
    elif altitude <= 32:
        lapse_rate = k_20_32
    elif altitude <= 47:
        lapse_rate = k_32_47
    else:
        lapse_rate = 0.0 # IDK about this one for now.
    return lapse_rate # K/km

def get_temp_from_altitude(altitude):
    """
    Returns the temperature at a given altitude.
    altitude [km]
    """
    # Get temps at layer boundaries
    T11 = T_sea + k_0_11 * (11 - 0)
    T20 = T11 + k_11_20 * (20 - 11)
    T32 = T20 + k_20_32 * (32 - 20)
    T47 = T32 + k_32_47 * (47 - 32)

    lapse_rate = get_lapse_rate(altitude)
    if altitude <0:
        raise ValueError("Altitude cannot be negative.")
    else:
        if altitude < 11:
            return T_sea + lapse_rate * (altitude - 0)
        elif altitude < 20:
            return T11 + lapse_rate * (altitude - 11)
        elif altitude < 32:
            return T20 + lapse_rate * (altitude - 20)
        elif altitude <= 47:
            return T32 + lapse_rate * (altitude - 32)
        else:
            raise NotImplementedError("Altitude > 47 km not handled yet.")


def get_altitude_reference(altitude):
    """
    Returns the reference altitude at the base of the layer containing the given altitude.
    altitude [km]
    """
    if altitude < 0:
        raise ValueError("Altitude cannot be negative.")
    for i, bound in enumerate(altitude_bounds):
        if altitude <= bound:
            return altitude_bounds[i-1] if i > 0 else altitude_bounds[0]
    return altitude_bounds[-1]


def get_temperature_reference(altitude):
    """
    Returns the reference temperature at the base of the layer containing the given altitude.
    altitude [km]
    """
    altitude_ref = get_altitude_reference(altitude) 
    return get_temp_from_altitude(altitude_ref)


def constant_temp_pressure(altitude, pressure_ref, specific_R):
    """
    Returns the pressure at a given altitude assuming a constant temperature.
    altitude, altitude_ref [km]
    temp_ref [K]
    pressure_ref [Pa]
    specific_R [J/(kg*K)]
    """
    # This will only be for 11-20km layer
    altitude_ref = get_altitude_reference(altitude)
    temp_ref     = get_temperature_reference(altitude)   # K
    dh = (altitude - altitude_ref) * 10**3            # m
    return pressure_ref * np.exp(-(g * dh) / (specific_R * temp_ref))  # Pa


def linear_temp_pressure(altitude, pressure_ref, specific_R):
    """
    Returns the pressure at a given altitude assuming a linear temperature gradient.
    altitude, altitude_ref [km]
    temp_ref [K]
    pressure_ref [Pa]
    specific_R [J/(kg*K)]
    """
    altitude_ref = get_altitude_reference(altitude)
    temp_ref     = get_temperature_reference(altitude) # K
    lapse_rate   = get_lapse_rate(altitude) * 10**(-3) # K/m  (from K/km)
    dh = (altitude - altitude_ref) * 10**3 # m (from km)
    return pressure_ref * ((temp_ref + lapse_rate * dh) / temp_ref) ** (-g / (specific_R * lapse_rate))


def get_pressure_from_altitude(altitude):
    """
    Returns the pressure at a given altitude.
    altitude [km]
    """
    if altitude < 0:
        raise ValueError("Altitude cannot be negative.")

    # Pressure references at boundaries 
    p0  = P_sea
    p11 = linear_temp_pressure(11.0, p0,  R_air)
    p20 = constant_temp_pressure(20.0, p11, R_air)
    p32 = linear_temp_pressure(32.0, p20, R_air)
    p47 = linear_temp_pressure(47.0, p32, R_air)

    if altitude <= 11.0:
        return linear_temp_pressure(altitude, p0,  R_air)
    elif altitude <= 20.0: # isothermal layer
        return constant_temp_pressure(altitude, p11, R_air)
    elif altitude <= 32.0:
        return linear_temp_pressure(altitude, p20, R_air)
    elif altitude <= 47.0:
        return linear_temp_pressure(altitude, p32, R_air)
    else:
        raise NotImplementedError("Altitude ≥ 47 km not handled yet.")
    
def get_altitude_from_temperature(temperature, tol=1e-6):
    """
    Return ALL altitude solutions h (in km) such that get_temp_from_altitude(h) == temperature

    Returns:
      List of dicts:
        - {"type": "point", "altitude_km": float}
        - {"type": "interval", "low_km": float, "high_km": float} This is just for the entire 11-20 isothermal layer
    """
    # Get temps at layer boundaries
    T0 = T_sea
    T11 = T0 + k_0_11 * (11.0 - 0.0)
    T20 = T11 + k_11_20 * (20.0 - 11.0) #isothermal layer. T20 = T11
    T32 = T20 + k_20_32 * (32.0 - 20.0)
    T47 = T32 + k_32_47 * (47.0 - 32.0)

    solutions = []

    # Check layer 1: 0 ≤ h ≤ 11 (linear)
    # T = T_0 + k_0_11 * (h - 0)
    h = (temperature - T0) / k_0_11
    if -tol <= h <= 11.0 + tol:
        h = float(max(0.0, min(11.0, h))) # Clamps to our layers just in case theres some weird numerical stuff goin on
        solutions.append({"type": "point", "altitude_km": h})
    
    # Check layer 2: 11 < h ≤ 20 (isothermal constant)
    if abs(temperature - T11) <= tol or abs(temperature - T20) <= tol:
        solutions.append({"type": "interval", "low_km": 11.0, "high_km": 20.0})

    # Check layer 3: 20 < h ≤ 32 (linear)
    # T = T_20 + k_20_32 * (h - 20)
    h = 20.0 + (temperature - T20) / k_20_32
    if (h > 20.0 + tol) and (h <= 32.0 + tol):
        h = float(min(32.0, h))
        solutions.append({"type": "point", "altitude_km": h})

    # Check layer 4: 32 < h ≤ 47 (linear)
    # T = T_32 + k_32_47 * (h - 32)
    h = 32.0 + (temperature - T32) / k_32_47
    if (h > 32.0 + tol) and (h <= 47.0 + tol):
        h = float(min(47.0, h))
        solutions.append({"type": "point", "altitude_km": h})

    # If nothing matched, return empty list (temperature outside modeled range)
    return solutions

