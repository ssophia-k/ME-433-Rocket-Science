import numpy as np
import matplotlib.pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.numerical_iterator import *

def get_P0_from_static(P, M, gamma=1.4):
    """Calculate Total Pressure from Static Pressure and Mach."""
    return P * (1 + (gamma - 1)/2 * M**2)**(gamma / (gamma - 1))

def get_T0_from_static(T, M, gamma=1.4):
    """Calculate Total Temp from Static Temp and Mach."""
    return T * (1 + (gamma - 1)/2 * M**2)


def get_area_from_mach(M, A_star, gamma=1.4):
    """
    Calculates Area A for a given Mach number using Isentropic relation.
    """
    term1 = 1 / M
    term2 = (2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * M**2)
    exponent = (gamma + 1) / (2 * (gamma - 1))
    return A_star * term1 * (term2 ** exponent)


def area_mach_relation(M, gamma):
    """
    Finds the ratio (A/A*)^2 via area mach relation.
    Inputs:
        Mach number (unitless)
        specific heat ratio gamma (unitless)
    Returns:
        (A/A*)^2 (unitless)
    """
    return (1.0/M**2) *  ( (2.0/(gamma+1.0))* (1.0 + ((gamma-1.0)/2.0) * M**2) ) ** ( (gamma+1.0)/ (gamma-1.0))

def inverse_area_mach_relation(A_ratio_sq, gamma):
    """
    Numerically solves for Mach number given the (A/A*)^2 ratio.
    There are two solutions given a ratio, so we will do so numerically.
    Inputs:
        (A/A*)^2 (unitless)
        specific heat ratio gamma (unitless)
    Returns:
        List of Mach numbers (unitless)
    """
    if gamma <= 1:
        raise ValueError("gamma must be > 1")
    if A_ratio_sq < 1:
        return [float('nan'), float('nan')]
    if abs(A_ratio_sq - 1.0) <= 1e-12:
        return [1.0, 1.0] # this is the choke point A*. numerical tolerance stuff

    def area_ratio_sq(M):
        return (1.0/M**2) * (
            (2.0/(gamma+1.0)) * (1.0 + 0.5*(gamma-1.0)*M**2)
        ) ** ((gamma+1.0)/(gamma-1.0))

    # subsonic: (0,1), supersonic: (1, 50]
    try:
        sub_sln = numerical_iterator(area_ratio_sq, 1e-12, 1.0 - 1e-12, A_ratio_sq, tol=1e-8, max_iter=10000)
    except Exception:
        sub_sln = float('nan')

    try:
        super_sln = numerical_iterator(area_ratio_sq, 1.0 + 1e-12, 50.0, A_ratio_sq, tol=1e-8, max_iter=10000)
    except Exception:
        super_sln = float('nan')

    return [sub_sln, super_sln]


