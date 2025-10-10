import numpy as np
from .constants import *
from .standard_atmosphere import get_temp_from_altitude, get_pressure_from_altitude

def P0_P(M, gamma):
    """
    Find stagnation pressure ratio via isentropic process
    Parameters:
        M: Mach number (unitless)
        gamma: ratio of specific heats (unitless)
    Returns:
        P0/P: stagnation pressure versus pressure ratio (unitless)
    """
    return (1 + (gamma-1)/2 * M**2) ** (gamma/(gamma-1))

def get_mean_free_path(altitude, diameter):
    """
    Returns the mean free path of air molecules at a given altitude.
    altitude [km]
    diameter [m] - diameter of the molecule
    """
    if altitude <0:
        raise ValueError("Altitude cannot be negative.")
    temp = get_temp_from_altitude(altitude)
    pressure = get_pressure_from_altitude(altitude)
    mean_free_path = (k_b * temp) / (np.sqrt(2) * np.pi * (diameter**2) * pressure)
    return mean_free_path # m

def get_knudsen_number(mean_free_path, characteristic_length):
    """
    Returns the Knudsen number.
    mean_free_path [m]
    characteristic_length [m]
    """
    if characteristic_length <= 0:
        raise ValueError("Characteristic length must be positive.")
    knudsen_number = mean_free_path / characteristic_length
    return knudsen_number # dimensionless