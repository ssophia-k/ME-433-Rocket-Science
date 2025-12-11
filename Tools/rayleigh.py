import numpy as np

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

def P_Pstar(M, gamma):
    return (1+gamma)/(1+ gamma*M**2)

def T_Tstar(M, gamma):
    return M**2 * (P_Pstar(M, gamma))**2

def P0_P0star(M, gamma):
    return P_Pstar(M, gamma) * ( (2 + (gamma-1)*M**2) / (gamma+1))**(gamma/ (gamma-1))

def T0_T0star(M, gamma):
    return ( ((gamma+1) * M**2) / (1 + gamma* M**2)**2) * (2 + (gamma-1) * M**2)

def get_q(T0_1, T0_2, cp):
    return cp * (T0_2-T0_1)