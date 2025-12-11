# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 11:42:17 2025

@author: Adin Sacho-Tanzer
"""

def flameholder(P_in, M_in, m_dot, gamma=1.4):
    P_out = P_in - P_in *0.81*gamma*M_in**2
    return P_out, M_in