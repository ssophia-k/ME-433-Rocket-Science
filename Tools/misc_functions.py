import numpy as np

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.constants import *
from Tools.standard_atmosphere import get_temp_from_altitude, get_pressure_from_altitude

def get_speed_of_sound(T, R = R_air, gamma = gamma_air):
    """
    Returns the speed of sound at a give temperature
    T [K] - temperature
    R [J/kg*K] - specific gas constant
    gamma [unitless] - ratio of specific heats
    Returns speed of sound in [m/s]
    """
    return np.sqrt(gamma * R * T)

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