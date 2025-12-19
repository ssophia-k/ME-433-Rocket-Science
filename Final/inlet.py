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
from Tools.isentropic import P0_P, T0_T

# Inlet should take in atmospheric conditions, desired m_dot, number of angles
# Should output P, M at input to diffuser (after NS), as well as P after all OSs but before NS for calculating external drag


def line_height(x, x0, y0, angle_deg):
    return y0 + np.tan(np.radians(angle_deg)) * (np.asarray(x) - x0)

def plot_field(X, Y, field, title, label):
    plt.figure(figsize=(6,4))
    plt.contourf(X, Y, field, levels=50)
    plt.colorbar(label=label)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.show()

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
        
    def plot(self, ax, structure_format="b", shock_format="r--"):
        for i in range(len(self.xs)-1):
            ax.plot([self.xs[i], self.xs[i+1]], [self.ys[i], self.ys[i+1]], structure_format)
        ax.scatter(self.x_lip, self.y_lip)
        for i in range(len(self.xs)-1):
            l = self.x_lip-self.xs[i]
            ax.plot([self.xs[i], self.x_lip], [self.ys[i], self.ys[i]+np.tan(np.deg2rad(self.betas[i]))*l], shock_format)
        
        # ax.plot([self.xs[-1], self.x_lip], [self.ys[-1], self.y_lip], shock_format, label="Shock")
        ax.legend()
        
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
    
    def compute_flow_fields(self, xs, ys, P_in, T_in, M_in):
        """
        Compute P, T, rho, M, P0, T0, s fields at a grid with coordinates xs, ys
        """
        rho_in = P_in / (R_air * T_in)
        cp = self.gamma * R_air / (self.gamma - 1)
    
        # Build mesh
        X, Y = np.meshgrid(xs, ys, indexing='xy')
    
        # Initialize ratio grids
        P_ratio   = np.ones_like(X, dtype=float)
        T_ratio   = np.ones_like(X, dtype=float)
        rho_ratio = np.ones_like(X, dtype=float)
        M_grid    = np.full_like(X, M_in, dtype=float)
    
        M_current = M_in
        n_shocks = len(self.turn_angles)
    
        for i in range(n_shocks):
            # Shock geometry
            xi, yi = self.xs[i], self.ys[i]
            theta  = self.turn_angles[i]
    
            # Shock relations
            beta, Pr, Tr, M_after, rhor = mach_function(M_current, self.gamma, theta)
            if i != 0:
                beta += self.location_angles[i-1]
    
            # Shock line
            Y_line = line_height(X, xi, yi, beta)
    
            # Downstream region
            mask = (Y < Y_line)
    
            # Apply jump conditions
            P_ratio[mask]   *= Pr
            T_ratio[mask]   *= Tr
            rho_ratio[mask] *= rhor
            M_grid[mask]     = M_after
    
            M_current = M_after
    
        # Primitive variables
        P_grid   = P_in   * P_ratio
        T_grid   = T_in   * T_ratio
        rho_grid = rho_in * rho_ratio
    
        # --- Stagnation quantities ---
        P0_grid = P_grid * P0_P(M_grid, self.gamma)
        T0_grid = T_grid * T0_T(M_grid, self.gamma)
    
        # --- Entropy relative to freestream ---
        s_grid = cp * np.log(T_grid / T_in) - R_air * np.log(P_grid / P_in)
    
        # Apply wall masking
        for i in range(len(self.xs) - 1):
            x1, y1 = self.xs[i],   self.ys[i]
            x2, y2 = self.xs[i+1], self.ys[i+1]
    
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            Y_wall = line_height(X, x1, y1, angle)
    
            x_min = min(x1, x2)
            x_max = max(x1, x2)
    
            wall_mask = (X >= x_min) & (X <= x_max) & (Y < Y_wall)
    
            P_grid[wall_mask]   = np.nan
            T_grid[wall_mask]   = np.nan
            rho_grid[wall_mask] = np.nan
            M_grid[wall_mask]   = np.nan
            P0_grid[wall_mask]  = np.nan
            T0_grid[wall_mask]  = np.nan
            s_grid[wall_mask]   = np.nan
    
        return P_grid, T_grid, rho_grid, M_grid, P0_grid, T0_grid, s_grid
    
    def get_1d_profiles(self, xs, P_in, T_in, M_in):
        """
        Compute mass-flux-averaged profiles P(x), T(x), rho(x), M(x),
        P0(x), T0(x), s(x)
        """
        ys = np.linspace(0, self.y_lip, num=200)
    
        (P_grid, T_grid, rho_grid, M_grid,
         P0_grid, T0_grid, s_grid) = self.compute_flow_fields(
            xs, ys, P_in, T_in, M_in
        )
    
        X, Y = np.meshgrid(xs, ys, indexing='xy')
        a_grid = get_speed_of_sound(T_grid)
    
        # Mass flux weights
        w = rho_grid * M_grid * a_grid
    
        # Output arrays
        P_profile   = np.zeros_like(xs)
        T_profile   = np.zeros_like(xs)
        rho_profile = np.zeros_like(xs)
        M_profile   = np.zeros_like(xs)
        P0_profile  = np.zeros_like(xs)
        T0_profile  = np.zeros_like(xs)
        s_profile   = np.zeros_like(xs)
    
        for j in range(len(xs)):
            if xs[j] > self.x_lip:
                vals = self.get_1d_profiles([self.x_lip], P_in, T_in, M_in)
                (P_profile[j], T_profile[j], rho_profile[j],
                 M_profile[j], P0_profile[j],
                 T0_profile[j], s_profile[j]) = [v[0] for v in vals]
                continue
    
            w_col = w[:, j]
            valid = ~np.isnan(w_col)
            wsum = np.sum(w_col[valid])
    
            if wsum == 0:
                P_profile[j]   = np.nan
                T_profile[j]   = np.nan
                rho_profile[j] = np.nan
                M_profile[j]   = np.nan
                P0_profile[j]  = np.nan
                T0_profile[j]  = np.nan
                s_profile[j]   = np.nan
                continue
    
            P_profile[j]   = np.sum(P_grid[:, j][valid]   * w_col[valid]) / wsum
            T_profile[j]   = np.sum(T_grid[:, j][valid]   * w_col[valid]) / wsum
            rho_profile[j] = np.sum(rho_grid[:, j][valid] * w_col[valid]) / wsum
            M_profile[j]   = np.sum(M_grid[:, j][valid]   * w_col[valid]) / wsum
            P0_profile[j]  = np.sum(P0_grid[:, j][valid]  * w_col[valid]) / wsum
            T0_profile[j]  = np.sum(T0_grid[:, j][valid]  * w_col[valid]) / wsum
            s_profile[j]   = np.sum(s_grid[:, j][valid]   * w_col[valid]) / wsum
    
        return (P_profile, T_profile, rho_profile, M_profile,
                P0_profile, T0_profile, s_profile)
        
            
if __name__ == "__main__":
    i = inlet(9112.32, 216.65, 3.25, 1, [10, 10, 10, 10])
    ax = plt.subplot()
    i.plot(ax)
    ax.set_aspect('equal')
    plt.show()
    
    for label, val in zip(["M_normal", "P_normal", "T_normal", "M_oblique", "P_oblique", "T_oblique"], i.output_properties(9112.32, 216.65, 3.25)):
        print(f"{label}: {val}")
    
    inlet_width = np.sqrt((i.xs[-1]-i.x_lip)**2+(i.ys[-1]-i.y_lip)**2)
    M, P, T, _, _, _ = i.output_properties(9112.32, 216.65, 2.75)
    rho = P/(R_air*T)
    a = get_speed_of_sound(T)
    print(f"m_dot at throat = {M*a*rho*inlet_width*1}")
    print(f"total pressure drag: {i.get_pressure_drag(9112, 216, 3)} N")
    print(f"inlet momentum flux: {i.get_inlet_momentum_flux(9112, 216, 3)} N")
    
    xs = np.linspace(0, 0.015, 500)
    ys = np.linspace(0, i.y_lip, 500)
    P_grid, T_grid, rho_grid, M_grid, P0_grid, T0_grid, s_grid = i.compute_flow_fields(xs, ys, 9112.32, 216.65, 3.25)

    # Create mesh for plotting
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    
    # Plot primitive fields
    plot_field(X, Y, P_grid,   "Pressure Field",        "P (Pa)")
    plot_field(X, Y, T_grid,   "Temperature Field",     "T")
    plot_field(X, Y, rho_grid, "Density Field",         "rho")
    plot_field(X, Y, M_grid,   "Mach Number Field",     "M")
    
    # Plot stagnation + entropy fields
    plot_field(X, Y, P0_grid,  "Stagnation Pressure Field",    "P₀")
    plot_field(X, Y, T0_grid,  "Stagnation Temperature Field", "T₀")
    plot_field(X, Y, s_grid,   "Entropy Field (Δs)",           "s")
    
    (P_profile, T_profile, rho_profile, M_profile, P0_profile, T0_profile, s_profile) =  i.get_1d_profiles(xs, 9112.32, 216.65, 3.25)

    plt.plot(xs, s_profile)
    plt.show()
    
