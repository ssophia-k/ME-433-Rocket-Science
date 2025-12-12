# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 18:13:39 2025

@author: Adin Sacho-Tanzer
"""

import numpy as np
from matplotlib import pyplot as plt
from inlet import inlet as Inlet

def circle_top(xs, x_c, y_c, r):
    return y_c+np.sqrt(r**2-(x_c-xs)**2)

def plot_top(inlet, x_offset, y_offset, x_end):
    """
    plots top surface
    Parameters
    inlet : inlet object
    x_offset : x offset
    y_offset: y offset
    x_end : x to end at
    Returns:
    inner_coords: list of (x, y) pairs for bottom surface
    outer_coords: list of (x, y) pairs for top surface
    """
    x_c = inlet.xs[-1]
    y_c = inlet.ys[-1]
    r = np.sqrt((inlet.xs[-1]-inlet.x_lip)**2+(inlet.ys[-1]-inlet.y_lip)**2)
    
    xs_circle = np.linspace(inlet.x_lip, x_c,)
    ys_circle = circle_top(xs_circle, x_c, y_c, r)
    
    slope = (inlet.ys[-1]-inlet.ys[-2])/(inlet.xs[-1]-inlet.xs[-2])
    
    # Lists to store coordinates
    inner_coords = []
    outer_coords = []
    
    # Inner plots
    # 1. Circle
    for x, y in zip(xs_circle + x_offset, ys_circle + y_offset):
        inner_coords.append((x, y))
    
    # 2. Line connecting last circle point to x_end
    for x, y in zip([xs_circle[-1]+x_offset, x_end+x_offset],
                    [ys_circle[-1]+y_offset, ys_circle[-1]+y_offset]):
        inner_coords.append((x, y))
    
    # Outer plot
    # Line from inlet to x_c to x_end
    x_vals_outer = [inlet.x_lip+x_offset, x_c+x_offset, x_end+x_offset]
    y_vals_outer = [
        inlet.y_lip+y_offset,
        inlet.y_lip + slope*(x_c - inlet.x_lip) + y_offset,
        inlet.y_lip + slope*(x_c - inlet.x_lip) + y_offset
    ]
    
    for x, y in zip(x_vals_outer, y_vals_outer):
        outer_coords.append((x, y))

    # ax.plot(xs_circle+x_offset, ys_circle+y_offset, inner_format)
    # ax.plot([xs_circle[-1]+x_offset, x_end+x_offset], [ys_circle[-1]+y_offset, ys_circle[-1]+y_offset], inner_format)
    # ax.plot([inlet.x_lip+x_offset, x_c+x_offset, x_end+x_offset], [inlet.y_lip+y_offset, inlet.y_lip+slope*(x_c-inlet.x_lip)+y_offset, inlet.y_lip+slope*(x_c-inlet.x_lip)+y_offset], outer_format)
    
    # For top profile back thickness
    top_profile_back_thickness = inner_coords[-1][1] - outer_coords[-1][1]
    outer_coords.append(inner_coords[-1])   # For plotting

    xs_top = [i[0] for i in outer_coords]
    ys_top = [i[1] for i in outer_coords]
    
    xs_bottom = [i[0] for i in inner_coords]
    ys_bottom = [i[1] for i in inner_coords]
    
    # ax.plot(xs_top, ys_top)
    # ax.plot(xs_bottom, ys_bottom)
    
    return inner_coords, outer_coords, top_profile_back_thickness
    
 
    
if __name__ == "__main__":
     P_atm = 9112.32  # Pa
     T_atm = 216.65  # K
     M_atms = np.linspace(2.75, 3.25)
     M_atms = [3.25]
    
     # Basic properties:
     m_dot = 10  # kg/s
     width = 1  # m
    
     # Inlet:
     M_max = M_atms[-1]
     turn_angles = [10, 10, 10]  # turn angles of inlet, degrees
     inlet = Inlet(P_atm, T_atm, M_max, m_dot, turn_angles, width=width)
    
     ax = plt.subplot()
     plot_top(ax, inlet, 0, 0, 1, 0.005)
     ax.set_aspect("equal")
     # plt.xlim(0.95, 1.05)
     plt.xlim(0.1, 0.2)
     plt.ylim(0.05, 0.1)
     plt.show()