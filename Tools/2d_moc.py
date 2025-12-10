import numpy as np

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.misc_functions import get_speed_of_sound
from Tools.constants import *
from Tools.expansion_fan import *
from Tools.numerical_iterator import *

def get_mu(M):
    return np.arcsin(1/M)

def get_slope(theta, M, plus):
    mu = get_mu(M)
    if plus == True:
        return np.tan(theta+mu)
    else:
        return np.tan(theta-mu)

def get_K(theta, M, gamma, plus):
    nu = nu_func(M, gamma)
    if plus == True:
        return (theta - nu)
    else:
        return (theta + nu)
    
def get_theta_from_K(K, M, gamma, plus):
    nu = nu_func(M, gamma)
    if plus == True:
        return(K + nu)
    else:
        return(K-nu)

def get_M_from_K(K, theta, gamma, plus):
    if plus == True:
        nu = (theta-K)
        M = inverse_nu_func(nu, gamma)
    else:
        nu = (K-theta)
        M = inverse_nu_func(nu, gamma)
    return M


def intersect_char(K_plus, K_minus, gamma):
    theta = 0.5 * (K_minus + K_plus)
    nu = 0.5 * (K_minus - K_plus)
    M = inverse_nu_func(nu, gamma)
    return theta, M


def point_in_triangle(p, a, b, c):
    """
    p, a, b, c are (x, y) tuples
    Returns True if p is inside triangle abc, else False.
    """

    (px, py) = p
    (ax, ay) = a
    (bx, by) = b
    (cx, cy) = c

    # Vectors
    v0 = (cx - ax, cy - ay)
    v1 = (bx - ax, by - ay)
    v2 = (px - ax, py - ay)

    # Dot products
    dot00 = v0[0]*v0[0] + v0[1]*v0[1]
    dot01 = v0[0]*v1[0] + v0[1]*v1[1]
    dot02 = v0[0]*v2[0] + v0[1]*v2[1]
    dot11 = v1[0]*v1[0] + v1[1]*v1[1]
    dot12 = v1[0]*v2[0] + v1[1]*v2[1]

    # Compute barycentric coordinates
    denom = dot00 * dot11 - dot01 * dot01
    if denom == 0:
        return False  # Degenerate triangle

    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom

    # Inside check
    return (u >= 0 - 1e-12) and (v >= 0 - 1e-12) and (u + v <= 1 + 1e-12)

