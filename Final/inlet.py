# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 11:42:36 2025

@author: Adin Sacho-Tanzer
"""

import numpy as np
from matplotlib import pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))

from Tools.constants import R_air
from Tools.misc_functions import get_speed_of_sound
from Tools.oblique_shock import mach_function

# Inlet should take in atmospheric conditions, desired m_dot, number of angles
# Should output P, M at input to diffuser (after NS), as well as P after all OSs but before NS for calculating external drag


class inlet:
    def __init__(self, P_atm, T_atm, M_max, m_dot, turn_angles, width=1, gamma=1.4):
        """
        Inlet object
        Parameters
        P_atm : atmospheric pressure, Pa
        T_atm : atmospheric temperature, K
        M_max : maximum mach number for which inlet will be designed
        m_dot : desired mass flow rate of nozzle
        turn_angles : list of turn angles in inlet in degrees. must be at least one
        width : width of nozzle in m. The default is 1.
        gamma : ratio of specific heats. The default is 1.4.

        """
        rho_atm = P_atm/(R_air*T_atm)
        a = get_speed_of_sound(T_atm)
        self.y_lip = m_dot/(rho_atm*M_max*a*width)
        
        Ms = []
        self.turn_angles = turn_angles
        self.location_angles = []
        for i in range(len(turn_angles)):
            self.location_angles.append(sum(turn_angles[:i+1]))
            
        self.xs = [0]
        self.ys = [0]
        beta_initial, _, _, M2, _ = mach_function(M_max, gamma, turn_angles[0])
        Ms.append(M2)

        self.x_lip = self.y_lip/np.tan(np.deg2rad(beta_initial))
        
        for i in range(1, len(turn_angles)):
            theta = turn_angles[i]
            beta, _, _, M, _ = mach_function(Ms[-1], gamma, theta)
            Ms.append(M)
            x = (-self.ys[-1]+self.y_lip+self.xs[-1]*np.tan(np.deg2rad(self.location_angles[i-1]))-self.x_lip*np.tan(np.deg2rad(beta)))/(np.tan(np.deg2rad(self.location_angles[i-1]))-np.tan(np.deg2rad(beta)))
            y = self.y_lip+(x-self.x_lip)*np.tan(np.deg2rad(beta))
            self.xs.append(x)
            self.ys.append(y)
        
    
    def plot(self, ax):
        length = self.x_lip-self.xs[-1]
        for i in range(len(self.xs)-1):
            ax.plot([self.xs[i], self.xs[i+1]], [self.ys[i], self.ys[i+1]])
        ax.plot([self.xs[-1], self.xs[-1]+length], [self.ys[-1], self.ys[-1]+np.tan(np.deg2rad(self.location_angles[-1]))*length])
        ax.scatter(self.x_lip, self.y_lip)
        
    def output_properties(self):
        pass
        
            
            
    
        
if __name__ == "__main__":
    i = inlet(101325, 300, 3.25, 1, [5, 5, 5, 5])
    ax = plt.subplot()
    i.plot(ax)
    plt.show()
        