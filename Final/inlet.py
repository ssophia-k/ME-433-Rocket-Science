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
from Tools.normal_shock import M2_from_normal_shock, P2_P1_from_normal_shock, T2_T1_from_normal_shock

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
        
        self.width = width
        self.gamma = gamma
        self.M_max = M_max
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
        
        self.betas = [beta_initial]
        
        for i in range(1, len(turn_angles)):
            theta = turn_angles[i]
            beta, _, _, M, _ = mach_function(Ms[-1], gamma, theta)
            beta += self.location_angles[i-1]
            self.betas.append(beta)
            Ms.append(M)
            x = (-self.ys[-1]+self.y_lip+self.xs[-1]*np.tan(np.deg2rad(self.location_angles[i-1]))-self.x_lip*np.tan(np.deg2rad(beta)))/(np.tan(np.deg2rad(self.location_angles[i-1]))-np.tan(np.deg2rad(beta)))
            y = self.y_lip+(x-self.x_lip)*np.tan(np.deg2rad(beta))
            self.xs.append(x)
            self.ys.append(y)
        
        # Add an extra final point which is the closest point of the final line to the lip point:
        theta = np.radians(self.location_angles[-1])
        dx, dy = np.cos(theta), np.sin(theta)
        t = (self.x_lip - self.xs[-1]) * dx + (self.y_lip - self.ys[-1]) * dy
        x_end = self.xs[-1] + t * dx
        y_end = self.ys[-1] + t * dy
        self.xs.append(x_end)
        self.ys.append(y_end)
        
    def plot(self, ax):
        for i in range(len(self.xs)-1):
            ax.plot([self.xs[i], self.xs[i+1]], [self.ys[i], self.ys[i+1]])
        ax.scatter(self.x_lip, self.y_lip)
        for i in range(len(self.xs)-1):
            l = self.x_lip-self.xs[i]
            ax.plot([self.xs[i], self.x_lip], [self.ys[i], self.ys[i]+np.tan(np.deg2rad(self.betas[i]))*l], "--")
        
        ax.plot([self.xs[-1], self.x_lip], [self.ys[-1], self.y_lip], "--")
        
    def output_properties(self, P_in, T_in, M_in):
        """
        Get properties coming out of inlet
        Parameters
        P_in : Pressure of input flow, Pa
        T_in : Temperature of input flow, K
        M_in : Mach number of input flow
        Returns
        M_normal : Mach number of output flow into throat
        P_normal : Pressure of output flow into throat, Pa
        T_normal : Temperature of output flow into throat, K
        M_oblique : Mach number of flow past lip surface
        P_oblique : Pressure of flow past lip surface, Pa
        T_oblique : Temperature of flow past lip surface, K
        """
        M = M_in
        P = P_in
        T = T_in
        for theta in self.turn_angles:
            beta, Pr, Tr, M, rhor = mach_function(M, self.gamma, theta)
            P *= Pr
            T *= Tr
        
        M_oblique, P_oblique, T_oblique = M, P, T  # these are the properties after all oblique shocks, before normal shock
        
        M_normal = M2_from_normal_shock(M_oblique, self.gamma)
        P_normal = P_oblique*P2_P1_from_normal_shock(M_oblique, self.gamma)
        T_normal = T_oblique*T2_T1_from_normal_shock(M_oblique, self.gamma)
        
        return M_normal, P_normal, T_normal, M_oblique, P_oblique, T_oblique
    
    def get_pressure_drag(self, P_in, T_in, M_in):
        """
        Determine pressure drag on inlet. This returns a positive number representing the sum of all forces into the inlet in the x-direction
        Parameters
        P_in : input pressure, Pa
        T_in : input temperature, K
        M_in : input mach number
        Returns
        total_drag : presure drag force, N
        """
        M = M_in
        P = P_in
        total_drag = 0
        
        # Drag on bottom faces:
        for i in range(len(self.turn_angles)):
            theta = self.turn_angles[i]
            location_angle = self.location_angles[i]
            beta, Pr, Tr, M, rhor = mach_function(M, self.gamma, theta)
            P *= Pr
            length = np.sqrt((self.xs[i+1]-self.xs[i])**2+(self.ys[i+1]-self.ys[i])**2)
            force = P*length*self.width
            total_drag += force*np.sin(np.deg2rad(location_angle))
        
        # Drag on throat area:
        throat_area = np.sqrt((self.xs[-1]-self.x_lip)**2+(self.ys[-1]-self.y_lip)**2)*self.width
        throat_angle = np.arctan((self.xs[-1]-self.x_lip)/(self.ys[-1]-self.y_lip))
        # If we say inlet CV is before NS:
        _, _, _, _, throat_pressure, _ = self.output_properties(P_in, T_in, M_in)
        # If we say inlet CV is after NS:
        # _, throat_pressure, _, _, _, _ = self.output_properties(P_in, T_in, M_in)
        total_drag += throat_area*throat_pressure*np.cos(throat_angle)
        
        # Drag on lip upper surface:
        lip_slope = (self.ys[-1]-self.ys[-2])/(self.xs[-1]-self.xs[-2])
        lip_angle = np.arctan(lip_slope)
        lip_length = np.sqrt((self.x_lip-self.xs[-1])**2+(lip_slope*(self.xs[-1]-self.x_lip))**2)
        if M_in == self.M_max:
            # If we say no OS forms at the lip:
            # P_lip = P_in
            # If we say an OS forms at the lip:
            _, Pr, _, _, _= mach_function(M_in, self.gamma, np.degrees(lip_angle))
            P_lip = P_in * Pr
        else:
            _, _, _, _, P_lip, _ = self.output_properties(P_in, T_in, M_in)
        
        total_drag += lip_length*self.width*P_lip*np.sin(lip_angle)
        
        return total_drag
    
    def get_inlet_momentum_flux(self, P_in, T_in, M_in):
        """
        Get momentum flux in the x-direction through inlet throat. This is a positive number, may have to be to be made negative
        Parameters
        P_in : input pressure, Pa
        T_in : input temperature, K
        M_in : input mach number
        Returns
        Momentum flux through inlet, N
        """
        # If inlet CV is before NS:
        _, _, _, M_inlet, P_inlet, T_inlet = self.output_properties(P_in, T_in, M_in)
        # If inlet CV is after NS:
        # M_inlet, P_inlet, T_inlet, _, _, _ = self.output_properties(P_in, T_in, M_in)
        
        rho_inlet = P_inlet/(R_air*T_inlet)
        a_inlet = get_speed_of_sound(T_inlet)
        throat_area = np.sqrt((self.xs[-1]-self.x_lip)**2+(self.ys[-1]-self.y_lip)**2)*self.width
        throat_angle = np.arctan((self.xs[-1]-self.x_lip)/(self.ys[-1]-self.y_lip))
        
        return rho_inlet*(a_inlet*M_inlet)**2*throat_area*np.cos(throat_angle)        
        
            
if __name__ == "__main__":
    i = inlet(9112.32, 216.65, 3.25, 1, [5, 5, 5])
    ax = plt.subplot()
    i.plot(ax)
    ax.set_aspect('equal')
    plt.show()
    
    for label, val in zip(["M_normal", "P_normal", "T_normal", "M_oblique", "P_oblique", "T_oblique"], i.output_properties(9112.32, 216.65, 3.25)):
        print(f"{label}: {val}")
    
    inlet_width = np.sqrt((i.xs[-1]-i.x_lip)**2+(i.ys[-1]-i.y_lip)**2)
    M, P, T, _, _, _ = i.output_properties(9112.32, 216.65, 3.25)
    rho = P/(R_air*T)
    a = get_speed_of_sound(T)
    print(f"m_dot at throat = {M*a*rho*inlet_width*1}")
    print(f"total pressure drag: {i.get_pressure_drag(9112, 216, 3)} N")
    print(f"inlet momentum flux: {i.get_inlet_momentum_flux(9112, 216, 3)} N")