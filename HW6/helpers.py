import numpy as np

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.misc_functions import get_speed_of_sound
from Tools.constants import *

def get_slope(u, a, plus):
    if plus:
        return u+a
    else:
        return u-a
    
def get_J(u, a, gamma, plus):
    if plus:
        return u + (2 * a) / (gamma-1)
    else:
        return u - (2 * a) / (gamma-1)
    
def get_a_from_J(J, u, gamma, plus):
    a_plus = ((J-u)*(gamma-1))/2
    return a_plus if plus else -a_plus

def get_u_from_J(J, a, gamma, plus):
    if plus:
        return J - (2*a)/(gamma-1)
    else:
        return J + (2*a)/(gamma-1)

    
def intersect_char(J_plus, J_minus, gamma):
    a = ((gamma-1)/4) * (J_plus - J_minus)
    u = (1/2) * (J_plus + J_minus)
    return a, u