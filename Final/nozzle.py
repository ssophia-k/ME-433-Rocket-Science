import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))
from converging_section import design_converging_section

gamma = 1.4
R = 287

def design_nozzle(P5, T5, M5, m_dot, P_exit, depth, n_characteristics=300):
    """
    Design supersonic nozzle using Method of Characteristics
    
    Inputs:
        P5: Static pressure at throat (Pa)
        T5: Static temperature at throat (K)
        M5: Mach number at throat (dimensionless) - should be 1.0
        m_dot: Mass flow rate (kg/s)
        P_exit: Target exit pressure (Pa)
        depth: Nozzle depth for 2D analysis (m)
        n_characteristics: Number of characteristics
    
    Outputs:
        P, T, M, m_dot, A, r, x: Arrays with axial distributions
    """
    
    # Helper Functions
    def bisection_method(function, value, tolerance, left_bound=0, right_bound=47000):
        a = left_bound
        b = right_bound
        fa = function(a) - value
        fb = function(b) - value

        if abs(fa) < 1e-9:
            return a
        if abs(fb) < 1e-9:
            return b

        if fa * fb > 0:
            raise ValueError("Balloon will never reach equilibrium")
        while (b - a) > tolerance:
            c = (a + b) / 2
            fc = function(c) - value
            if abs(fc) < 1e-9:
                return c
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        return (a + b) / 2

    def prandtl_meyer(M, gamma):
        """Calculate Prandtl-Meyer angle
        Inputs:
            M: Mach number (dimensionless)
            gamma: ratio of specific heats (dimensionless)
        
        Outputs:
            nu: Prandtl-Meyer angle (radians)
        """
        if M < 1:
            return 0
        nu = np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1))) - np.arctan(np.sqrt(M**2 - 1))
        return nu

    def inverse_PM(nu_target, gamma):
        """Find Mach from Prandtl-Meyer angle using bisection
        Inputs:
            nu_target: Prandtl-Meyer angle (radians)
            gamma: ratio of specific heats (dimensionless)
        
        Outputs:
            M: Mach number (dimensionless)
        """
        def PM_function(M):
            return prandtl_meyer(M, gamma)
        
        M = bisection_method(PM_function, nu_target, tolerance=1e-6, left_bound=1.0, right_bound=20.0)
        return M

    def mach_angle(M):
        return np.arcsin(1.0 / M)

    # Calculate stagnation properties and throat geometry from inputs
    P0 = P5 * (1 + (gamma-1)/2 * M5**2)**(gamma/(gamma-1))
    T0 = T5 * (1 + (gamma-1)/2 * M5**2)
    
    M_exit = np.sqrt(2/(gamma-1) * ((P0/P_exit)**((gamma-1)/gamma) - 1))

    rho5 = P5 / (R * T5)
    a5 = np.sqrt(gamma * R * T5)
    u5 = M5 * a5
    A_throat = m_dot / (rho5 * u5)
    height_throat = A_throat / (depth)
    
    M_throat = M5
    x_corner = 0.0
    y_corner = height_throat
    y_centerline = 0.0

    # Assumptions/Limitations:
    #   Inviscid, adiabatic flow
    #   Steady
    #   Air, gamma = 1.4
    #   Symmetric about the center line

    # Incident Waves (Expansion Fan)
    theta_head = 0.0
    nu_head = 0.0
    M_head = M_throat
    mu_head = mach_angle(M_head)

    nu_exit = prandtl_meyer(M_exit, gamma)
    theta_max = nu_exit / 2
    theta_tail = theta_max
    nu_tail = theta_max
    M_tail = inverse_PM(nu_tail, gamma)
    mu_tail = mach_angle(M_tail)

    # Generate C- Characteristics
    c_minus_chars = []
    for i in range(1, n_characteristics):
        theta = theta_head + i * (theta_tail - theta_head) / (n_characteristics - 1)
        nu = theta
        
        M_char = inverse_PM(nu, gamma)
        mu_char = mach_angle(M_char)
        char_angle = theta - mu_char
        K_minus = theta + nu
        
        c_minus_chars.append({
            'theta': theta,
            'nu': nu,
            'M': M_char,
            'K_minus': K_minus,
            'char_angle': char_angle,
            'path': [(x_corner, y_corner)],
            'properties': [{
                'M': M_char,
                'p_p0': 1 / ((1 + (gamma-1)/2 * M_char**2)**(gamma/(gamma-1))),
                'T_T0': 1 / (1 + (gamma-1)/2 * M_char**2),
                'theta': theta,
                'nu': nu,
                'K_minus': K_minus
            }]
        })

    c_plus_chars = []
    wall_points = [(x_corner, y_corner)]
    node_count = 0

    # Layer by Layer
    for layer in range(len(c_minus_chars)):
        c_minus = c_minus_chars[layer]

        if layer == 0:
            c_minus_centerline_angle = c_minus['char_angle']

        x_last, y_last = c_minus['path'][-1]
        x_centerline = x_last + (y_centerline - y_last) / np.tan(c_minus_centerline_angle)
        c_minus['path'].append((x_centerline, y_centerline))
        c_minus['properties'].append(c_minus['properties'][-1].copy())

        node_count += 1

        # Center Line Interactions
        K_minus_at_centerline = c_minus['K_minus']
        theta_centerline = 0
        nu_centerline = K_minus_at_centerline - theta_centerline
        K_plus_reflected = theta_centerline - nu_centerline

        M_centerline = inverse_PM(abs(nu_centerline), gamma) if abs(nu_centerline) > 0 else 1.0
        mu_centerline = mach_angle(M_centerline) if M_centerline > 1 else np.pi/2
        c_plus_angle = theta_centerline + mu_centerline

        c_plus_path = [(x_centerline, y_centerline)]
        c_plus_properties = [{
            'M': M_centerline,
            'p_p0': 1 / ((1 + (gamma-1)/2 * M_centerline**2)**(gamma/(gamma-1))),
            'T_T0': 1 / (1 + (gamma-1)/2 * M_centerline**2),
            'theta': theta_centerline,
            'nu': nu_centerline,
            'K_plus': K_plus_reflected
        }]

        c_plus_K_plus = K_plus_reflected
        c_plus_theta = theta_centerline

        for j in range(layer + 1, len(c_minus_chars)):
            c_minus_next = c_minus_chars[j]

            x_c_plus, y_c_plus = c_plus_path[-1]
            x_c_minus, y_c_minus = c_minus_next['path'][-1]

            tan_plus = np.tan(c_plus_angle)
            tan_minus = np.tan(c_minus_next['char_angle'])

            x_int = (y_c_minus - y_c_plus + tan_plus*x_c_plus - tan_minus*x_c_minus) / (tan_plus - tan_minus)
            y_int = y_c_plus + tan_plus * (x_int - x_c_plus)

            c_plus_path.append((x_int, y_int))
            c_minus_next['path'].append((x_int, y_int))

            node_count += 1

            K_minus_int = c_minus_next['K_minus']
            theta_int = (c_plus_K_plus + K_minus_int) / 2
            nu_int = (K_minus_int - c_plus_K_plus) / 2 

            M_int = inverse_PM(abs(nu_int), gamma) if abs(nu_int) > 0 else 1.0
            mu_int = mach_angle(M_int) if M_int > 1 else np.pi/2

            intersection_props = {
                'M': M_int,
                'p_p0': 1 / ((1 + (gamma-1)/2 * M_int**2)**(gamma/(gamma-1))),
                'T_T0': 1 / (1 + (gamma-1)/2 * M_int**2),
                'theta': theta_int,
                'nu': nu_int,
                'K_plus': theta_int - nu_int,
                'K_minus': K_minus_int
            }
            
            c_plus_properties.append(intersection_props.copy())
            c_minus_next['properties'].append(intersection_props.copy())

            c_plus_angle = theta_int + mu_int
            c_minus_next['char_angle'] = theta_int - mu_int
            c_plus_theta = theta_int
            c_plus_K_plus = theta_int - nu_int

            if j == layer + 1:
                c_minus_centerline_angle = c_minus_next['char_angle']
        
        # Forming Nozzle Wall
        x_last, y_last = c_plus_path[-1]
        x_wall_prev, y_wall_prev = wall_points[-1]
        
        if layer == 0:
            theta_prev = theta_max
        else:
            theta_prev = c_plus_chars[-1]['theta_wall']
        
        theta_wall_avg = (theta_prev + c_plus_theta) / 2
        tan_wall = np.tan(theta_wall_avg)
        tan_plus = np.tan(c_plus_angle)
        
        x_wall = (y_wall_prev - y_last + tan_plus*x_last - tan_wall*x_wall_prev) / (tan_plus - tan_wall)
        y_wall = y_wall_prev + tan_wall * (x_wall - x_wall_prev)
        
        c_plus_path.append((x_wall, y_wall))
        c_plus_properties.append(c_plus_properties[-1].copy())
        c_plus_chars.append({
            'path': c_plus_path,
            'properties': c_plus_properties,
            'theta_wall': c_plus_theta
        })

        wall_points.append((x_wall, y_wall))

    # Extract wall coordinates
    wall_x = np.array([p[0] for p in wall_points])
    wall_y = np.array([p[1] for p in wall_points])
    
    # Calculate stagnation properties
    P0 = P5 * (1 + (gamma-1)/2 * M5**2)**(gamma/(gamma-1))
    T0 = T5 * (1 + (gamma-1)/2 * M5**2)
    
    # Create uniform x-spacing across nozzle
    x_min = min(wall_x)
    x_max = max(wall_x)
    n_samples = len(wall_x)
    x_uniform = np.linspace(x_min, x_max, n_samples)
    
    # Interpolate wall position at each x
    wall_y_interp = np.interp(x_uniform, wall_x, wall_y)
    
    # Initialize lists
    M_mass_avg = []
    p_mass_avg = []
    T_mass_avg = []
    
    # Loop through each x-location
    for idx, x_sample in enumerate(x_uniform):
        y_wall_at_x = wall_y_interp[idx]
        intersections = []
        
        # Check for any C+ characteristic intersections
        for layer, c_plus in enumerate(c_plus_chars):
            path = c_plus['path']
            properties = c_plus['properties']
            
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i+1]
                
                if min(x1, x2) <= x_sample <= max(x1, x2):
                    if abs(x2 - x1) > 1e-9:
                        t = (x_sample - x1) / (x2 - x1)
                        y_at_x = y1 + t * (y2 - y1)
                    else:
                        y_at_x = y1
                    
                    # Compute local density ratio and mass flux at intersection
                    props = properties[i]
                    rho_rho0 = (1 + (gamma-1)/2 * props['M']**2)**(-1/(gamma-1))
                    u_a0 = props['M'] * np.sqrt(props['T_T0'])
                    rho_u = rho_rho0 * u_a0
                    
                    intersections.append({
                        'y': y_at_x,
                        'M': props['M'],
                        'p_p0': props['p_p0'],
                        'T_T0': props['T_T0'],
                        'rho_u': rho_u  # Store rho_u
                    })
                    break
        
        # Check for any C- characteristic intersections
        for layer, c_minus in enumerate(c_minus_chars):
            path = c_minus['path']
            properties = c_minus['properties']
            
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i+1]
                
                if min(x1, x2) <= x_sample <= max(x1, x2):
                    if abs(x2 - x1) > 1e-9:
                        t = (x_sample - x1) / (x2 - x1)
                        y_at_x = y1 + t * (y2 - y1)
                    else:
                        y_at_x = y1
                    
                    # Compute local density ratio and mass flux at intersection
                    props = properties[i]
                    rho_rho0 = (1 + (gamma-1)/2 * props['M']**2)**(-1/(gamma-1))
                    u_a0 = props['M'] * np.sqrt(props['T_T0'])
                    rho_u = rho_rho0 * u_a0
                    
                    intersections.append({
                        'y': y_at_x,
                        'M': props['M'],
                        'p_p0': props['p_p0'],
                        'T_T0': props['T_T0'],
                        'rho_u': rho_u  # Store rho_u
                    })
                    break

        # If no intersections, use throat values
        if len(intersections) == 0:
            M_mass_avg.append(M_throat)
            p_mass_avg.append(P0 / ((1 + (gamma-1)/2 * M_throat**2)**(gamma/(gamma-1))))
            T_mass_avg.append(T0 / (1 + (gamma-1)/2 * M_throat**2))
            continue
        
        # Sort by y for integration
        intersections.sort(key=lambda p: p['y'])
        
        # Initialize sums
        M_mass_sum = 0
        p_mass_sum = 0
        T_mass_sum = 0
        rho_u_sum = 0
        y_prev = 0
        
        # Integrate
        for i, intersection in enumerate(intersections):
            y_curr = intersection['y']
            dy = y_curr - y_prev
            
            if i == 0:
                M_val = intersection['M']
                p_val = intersection['p_p0']
                T_val = intersection['T_T0']
                rho_u_val = intersection['rho_u']
            else:
                # Average between consecutive intersections
                M_val = (intersections[i-1]['M'] + intersection['M']) / 2
                p_val = (intersections[i-1]['p_p0'] + intersection['p_p0']) / 2
                T_val = (intersections[i-1]['T_T0'] + intersection['T_T0']) / 2
                rho_u_val = (intersections[i-1]['rho_u'] + intersection['rho_u']) / 2
            
            # Mass-flux weighted sums
            M_mass_sum += M_val * rho_u_val * dy
            p_mass_sum += p_val * rho_u_val * dy
            T_mass_sum += T_val * rho_u_val * dy
            rho_u_sum += rho_u_val * dy
            y_prev = y_curr
        
        # Add last region to reach the top of nozzle wall
        y_last_intersection = intersections[-1]['y']
        if y_last_intersection < y_wall_at_x:
            dy = y_wall_at_x - y_last_intersection
            M_val = intersections[-1]['M']
            p_val = intersections[-1]['p_p0']
            T_val = intersections[-1]['T_T0']
            rho_u_val = intersections[-1]['rho_u']
            
            M_mass_sum += M_val * rho_u_val * dy
            p_mass_sum += p_val * rho_u_val * dy
            T_mass_sum += T_val * rho_u_val * dy
            rho_u_sum += rho_u_val * dy
        
        # Compute mass-flux average
        M_mass_avg.append(M_mass_sum / rho_u_sum)
        p_mass_avg.append(P0 * (p_mass_sum / rho_u_sum))
        T_mass_avg.append(T0 * (T_mass_sum / rho_u_sum))

    M = np.array(M_mass_avg)
    P = np.array(p_mass_avg)
    T = np.array(T_mass_avg)

    # Calculate area at each point
    h = wall_y_interp
    A = h * depth
    x = x_uniform

    return P, T, M, m_dot, A, h, x, c_plus_chars, c_minus_chars

def analyze_nozzle_moc(x6s, h6s, P5, T5, M5, depth, n_characteristics=300):
    """
    Analyze nozzle with quasi-1D isentropic flow
    
    Inputs:
        h6s: List of heights of nozzle design (m)
        x6s: List of x-coordinates of nozzle design (m)
        P5: Static pressure at section 5 (Pa)
        T5: Static temperature at section 5 (K)
        M5: Mach number at section 5 (dimensionless)
        depth: Physical depth of nozzle (m)
    
    Outputs:
        P, T, M: Arrays with axial distributions
    """

    # Helper Functions
    def bisection_method(function, value, tolerance, left_bound=0, right_bound=47000):
        a = left_bound
        b = right_bound
        fa = function(a) - value
        fb = function(b) - value

        if abs(fa) < 1e-9:
            return a
        if abs(fb) < 1e-9:
            return b

        if fa * fb > 0:
            raise ValueError("Balloon will never reach equilibrium")
        while (b - a) > tolerance:
            c = (a + b) / 2
            fc = function(c) - value
            if abs(fc) < 1e-9:
                return c
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        return (a + b) / 2

    def prandtl_meyer(M, gamma):
        if M < 1:
            return 0
        nu = np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1))) - np.arctan(np.sqrt(M**2 - 1))
        return nu

    def inverse_PM(nu_target, gamma):
        def PM_function(M):
            return prandtl_meyer(M, gamma)
        M = bisection_method(PM_function, nu_target, tolerance=1e-6, left_bound=1.0, right_bound=20.0)
        return M

    def mach_angle(M):
        return np.arcsin(1.0 / M)
    
    # Stagnation properties from static
    P0 = P5 * (1 + (gamma-1)/2 * M5**2)**(gamma/(gamma-1))
    T0 = T5 * (1 + (gamma-1)/2 * M5**2)
    
    # Estimate exit Mach from given wall geometry
    A_throat = h6s.iloc[0] * depth
    A_exit = h6s.iloc[-1] * depth
    area_ratio = A_exit / A_throat
    
    def area_mach_function(M):
        return np.sqrt((1/M**2) * ((2/(gamma+1))*(1 + (gamma-1)/2 * M**2))**((gamma+1)/(gamma-1)))
    
    M_exit = bisection_method(area_mach_function, area_ratio, 1e-9, 1.001, 10.0)
    
    # MOC setup
    height_throat = h6s.iloc[0]
    M_throat = M5
    x_corner = x6s.iloc[0]
    y_corner = height_throat
    y_centerline = 0.0
    
    theta_head = 0.0
    nu_head = 0.0
    
    nu_exit = prandtl_meyer(M_exit, gamma)
    theta_max = nu_exit / 2
    theta_tail = theta_max
    
    # Generate C- Characteristics
    c_minus_chars = []
    for i in range(1, n_characteristics):
        theta = theta_head + i * (theta_tail - theta_head) / (n_characteristics - 1)
        nu = theta
        
        M_char = inverse_PM(nu, gamma)
        mu_char = mach_angle(M_char)
        char_angle = theta - mu_char
        K_minus = theta + nu
        
        c_minus_chars.append({
            'theta': theta,
            'nu': nu,
            'M': M_char,
            'K_minus': K_minus,
            'char_angle': char_angle,
            'path': [(x_corner, y_corner)],
            'properties': [{
                'M': M_char,
                'p_p0': 1 / ((1 + (gamma-1)/2 * M_char**2)**(gamma/(gamma-1))),
                'T_T0': 1 / (1 + (gamma-1)/2 * M_char**2),
                'theta': theta,
                'nu': nu,
                'K_minus': K_minus
            }]
        })
    
    c_plus_chars = []
    
    # Layer by Layer
    for layer in range(len(c_minus_chars)):
        c_minus = c_minus_chars[layer]
        
        if layer == 0:
            c_minus_centerline_angle = c_minus['char_angle']
        
        x_last, y_last = c_minus['path'][-1]
        x_centerline = x_last + (y_centerline - y_last) / np.tan(c_minus_centerline_angle)
        c_minus['path'].append((x_centerline, y_centerline))
        c_minus['properties'].append(c_minus['properties'][-1].copy())
        
        # Center Line Interactions
        K_minus_at_centerline = c_minus['K_minus']
        theta_centerline = 0
        nu_centerline = K_minus_at_centerline - theta_centerline
        K_plus_reflected = theta_centerline - nu_centerline
        
        M_centerline = inverse_PM(abs(nu_centerline), gamma) if abs(nu_centerline) > 0 else 1.0
        mu_centerline = mach_angle(M_centerline) if M_centerline > 1 else np.pi/2
        c_plus_angle = theta_centerline + mu_centerline
        
        c_plus_path = [(x_centerline, y_centerline)]
        c_plus_properties = [{
            'M': M_centerline,
            'p_p0': 1 / ((1 + (gamma-1)/2 * M_centerline**2)**(gamma/(gamma-1))),
            'T_T0': 1 / (1 + (gamma-1)/2 * M_centerline**2),
            'theta': theta_centerline,
            'nu': nu_centerline,
            'K_plus': K_plus_reflected
        }]
        
        c_plus_K_plus = K_plus_reflected
        c_plus_theta = theta_centerline
        
        for j in range(layer + 1, len(c_minus_chars)):
            c_minus_next = c_minus_chars[j]
            
            x_c_plus, y_c_plus = c_plus_path[-1]
            x_c_minus, y_c_minus = c_minus_next['path'][-1]
            
            tan_plus = np.tan(c_plus_angle)
            tan_minus = np.tan(c_minus_next['char_angle'])
            
            x_int = (y_c_minus - y_c_plus + tan_plus*x_c_plus - tan_minus*x_c_minus) / (tan_plus - tan_minus)
            y_int = y_c_plus + tan_plus * (x_int - x_c_plus)
            
            c_plus_path.append((x_int, y_int))
            c_minus_next['path'].append((x_int, y_int))
            
            K_minus_int = c_minus_next['K_minus']
            theta_int = (c_plus_K_plus + K_minus_int) / 2
            nu_int = (K_minus_int - c_plus_K_plus) / 2
            
            M_int = inverse_PM(abs(nu_int), gamma) if abs(nu_int) > 0 else 1.0
            mu_int = mach_angle(M_int) if M_int > 1 else np.pi/2
            
            intersection_props = {
                'M': M_int,
                'p_p0': 1 / ((1 + (gamma-1)/2 * M_int**2)**(gamma/(gamma-1))),
                'T_T0': 1 / (1 + (gamma-1)/2 * M_int**2),
                'theta': theta_int,
                'nu': nu_int,
                'K_plus': theta_int - nu_int,
                'K_minus': K_minus_int
            }
            
            c_plus_properties.append(intersection_props.copy())
            c_minus_next['properties'].append(intersection_props.copy())
            
            c_plus_angle = theta_int + mu_int
            c_minus_next['char_angle'] = theta_int - mu_int
            c_plus_theta = theta_int
            c_plus_K_plus = theta_int - nu_int
            
            if j == layer + 1:
                c_minus_centerline_angle = c_minus_next['char_angle']
        
        # Extend C+ to hit the given wall
        x_last, y_last = c_plus_path[-1]
        tan_plus = np.tan(c_plus_angle)
        
        # Find intersection with wall segments
        x_wall_hit = None
        y_wall_hit = None
        
        for i in range(len(x6s) - 1):
            x1, x2 = x6s.iloc[i], x6s.iloc[i+1]
            y1, y2 = h6s.iloc[i], h6s.iloc[i+1]
            
            if x2 < x_last:
                continue
            
            if abs(x2 - x1) < 1e-9:
                continue
            
            tan_wall = (y2 - y1) / (x2 - x1)
            
            if abs(tan_plus - tan_wall) < 1e-6:
                continue
            
            x_int = (y1 - y_last + tan_plus*x_last - tan_wall*x1) / (tan_plus - tan_wall)
            
            if x_int >= x1 and x_int <= x2 and x_int > x_last:
                y_int = y_last + tan_plus * (x_int - x_last)
                y_wall_check = y1 + tan_wall * (x_int - x1)
                
                if abs(y_int - y_wall_check) < 0.1:
                    if x_wall_hit is None or x_int < x_wall_hit:
                        x_wall_hit = x_int
                        y_wall_hit = y_int
        
        if x_wall_hit is not None:
            c_plus_path.append((x_wall_hit, y_wall_hit))
            c_plus_properties.append(c_plus_properties[-1].copy())
        
        c_plus_chars.append({
            'path': c_plus_path,
            'properties': c_plus_properties
        })
    
    # Create uniform x-spacing across nozzle
    x_max_moc = max([p[0] for char in c_plus_chars for p in char['path']])
    x_min_moc = min([p[0] for char in c_plus_chars for p in char['path']])
    n_samples = len(x6s)
    x_uniform = np.linspace(x_min_moc, x_max_moc, n_samples)
    
    # Interpolate wall position at each x
    wall_y_interp = np.interp(x_uniform, x6s, h6s)
    
    # Initialize lists
    M_mass_avg = []
    p_mass_avg = []
    T_mass_avg = []
    
    # Loop through each x-location
    for idx, x_sample in enumerate(x_uniform):
        y_wall_at_x = wall_y_interp[idx]
        intersections = []
        
        # Check for any C+ characteristic intersections
        for layer, c_plus in enumerate(c_plus_chars):
            path = c_plus['path']
            properties = c_plus['properties']
            
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i+1]
                
                if min(x1, x2) <= x_sample <= max(x1, x2):
                    if abs(x2 - x1) > 1e-9:
                        t = (x_sample - x1) / (x2 - x1)
                        y_at_x = y1 + t * (y2 - y1)
                    else:
                        y_at_x = y1
                    
                    props = properties[i]
                    rho_rho0 = (1 + (gamma-1)/2 * props['M']**2)**(-1/(gamma-1))
                    u_a0 = props['M'] * np.sqrt(props['T_T0'])
                    rho_u = rho_rho0 * u_a0
                    
                    intersections.append({
                        'y': y_at_x,
                        'M': props['M'],
                        'p_p0': props['p_p0'],
                        'T_T0': props['T_T0'],
                        'rho_u': rho_u
                    })
                    break
        
        # Check for any C- characteristic intersections
        for layer, c_minus in enumerate(c_minus_chars):
            path = c_minus['path']
            properties = c_minus['properties']
            
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i+1]
                
                if min(x1, x2) <= x_sample <= max(x1, x2):
                    if abs(x2 - x1) > 1e-9:
                        t = (x_sample - x1) / (x2 - x1)
                        y_at_x = y1 + t * (y2 - y1)
                    else:
                        y_at_x = y1
                    
                    props = properties[i]
                    rho_rho0 = (1 + (gamma-1)/2 * props['M']**2)**(-1/(gamma-1))
                    u_a0 = props['M'] * np.sqrt(props['T_T0'])
                    rho_u = rho_rho0 * u_a0
                    
                    intersections.append({
                        'y': y_at_x,
                        'M': props['M'],
                        'p_p0': props['p_p0'],
                        'T_T0': props['T_T0'],
                        'rho_u': rho_u
                    })
                    break
        
        if len(intersections) == 0:
            M_mass_avg.append(M_throat)
            p_mass_avg.append(P0 / ((1 + (gamma-1)/2 * M_throat**2)**(gamma/(gamma-1))))
            T_mass_avg.append(T0 / (1 + (gamma-1)/2 * M_throat**2))
            continue
        
        intersections.sort(key=lambda p: p['y'])
        
        M_mass_sum = 0
        # p_mass_sum = 0
        # T_mass_sum = 0
        rho_u_sum = 0
        y_prev = 0
        
        for i, intersection in enumerate(intersections):
            y_curr = intersection['y']
            dy = y_curr - y_prev
            
            if i == 0:
                M_val = intersection['M']
                # p_val = intersection['p_p0']
                # T_val = intersection['T_T0']
                rho_u_val = intersection['rho_u']
            else:
                M_val = (intersections[i-1]['M'] + intersection['M']) / 2
                # p_val = (intersections[i-1]['p_p0'] + intersection['p_p0']) / 2
                # T_val = (intersections[i-1]['T_T0'] + intersection['T_T0']) / 2
                rho_u_val = (intersections[i-1]['rho_u'] + intersection['rho_u']) / 2
            
            M_mass_sum += M_val * rho_u_val * dy
            # p_mass_sum += p_val * rho_u_val * dy
            # T_mass_sum += T_val * rho_u_val * dy
            rho_u_sum += rho_u_val * dy
            y_prev = y_curr
        
        y_last_intersection = intersections[-1]['y']
        if y_last_intersection < y_wall_at_x:
            dy = y_wall_at_x - y_last_intersection
            M_val = intersections[-1]['M']
            # p_val = intersections[-1]['p_p0']
            # T_val = intersections[-1]['T_T0']
            rho_u_val = intersections[-1]['rho_u']
            
            M_mass_sum += M_val * rho_u_val * dy
            # p_mass_sum += p_val * rho_u_val * dy
            # T_mass_sum += T_val * rho_u_val * dy
            rho_u_sum += rho_u_val * dy
        
        # M_mass_avg.append(M_mass_sum / rho_u_sum)
        # p_mass_avg.append(P0 * (p_mass_sum / rho_u_sum))
        # T_mass_avg.append(T0 * (T_mass_sum / rho_u_sum))

        # Averaged Mach number
        M_avg = M_mass_sum / rho_u_sum
        
        # Calculate P and T to be consistent with constant P0, T0
        P_consistent = P0 / (1 + (gamma-1)/2 * M_avg**2)**(gamma/(gamma-1))
        T_consistent = T0 / (1 + (gamma-1)/2 * M_avg**2)
        
        M_mass_avg.append(M_avg)
        p_mass_avg.append(P_consistent)
        T_mass_avg.append(T_consistent)
    
    Ms = np.array(M_mass_avg)
    Ps = np.array(p_mass_avg)
    Ts = np.array(T_mass_avg)

    return Ps, Ts, Ms, c_plus_chars, c_minus_chars

def analyze_nozzle_1d(h6s, P5, T5, M5, depth):
    """
    Analyze nozzle with quasi-1D isentropic flow
    
    Inputs:
        h6s: List of heights of nozzle design (m)
        P5: Static pressure at section 5 (Pa)
        T5: Static temperature at section 5 (K)
        M5: Mach number at section 5 (dimensionless)
        depth: Physical depth of nozzle (m)
    
    Outputs:
        P, T, M: Arrays with axial distributions
    """

    """
    Analyze nozzle with quasi-1D isentropic flow
    
    Inputs:
        h6s: List of heights of nozzle design (m)
        P5: Static pressure at section 5 (Pa)
        T5: Static temperature at section 5 (K)
        M5: Mach number at section 5 (dimensionless)
        depth: Physical depth of nozzle (m)
    
    Outputs:
        P, T, M: Arrays with axial distributions
    """

    # Helper Functions
    def bisection_method(function, value, tolerance, left_bound=0, right_bound=47000):
        a = left_bound
        b = right_bound
        fa = function(a) - value
        fb = function(b) - value

        if abs(fa) < 1e-9:
            return a
        if abs(fb) < 1e-9:
            return b

        if fa * fb > 0:
            raise ValueError("Balloon will never reach equilibrium")
        while (b - a) > tolerance:
            c = (a + b) / 2
            fc = function(c) - value
            if abs(fc) < 1e-9:
                return c
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        return (a + b) / 2

    def area_mach_function(M):
        """
        Inputs:
            M: Mach number (dimensionless)

        Outputs:
            A/A*: Area ratio (dimensionless)
        """
        return np.sqrt((1/M**2) * ((2/(gamma+1))*(1 + (gamma-1)/2 * M**2))**((gamma+1)/(gamma-1)))
    
    # Stagnation properties from static
    P0 = P5 * (1 + (gamma-1)/2 * M5**2)**(gamma/(gamma-1))
    T0 = T5 * (1 + (gamma-1)/2 * M5**2)
    # Throat Area
    A_throat = h6s.iloc[0] * depth

    # Area ratio at each wall point
    A = h6s * depth
    A_ratio = A / A_throat

    # Mach at each point
    Ms = []
    for i, ar in enumerate(A_ratio):
        if i == 0:
            M = M5  # Throat
        else:
            M = bisection_method(area_mach_function, ar, 1e-9, 1.001, 10.0)
        Ms.append(M)

    Ms = np.array(Ms)

    # Pressure and temperature ratios
    p_p0 = 1 / ((1 + (gamma-1)/2 * Ms**2)**(gamma/(gamma-1)))
    T_T0 = 1 / (1 + (gamma-1)/2 * Ms**2)
    Ps = p_p0 * P0
    Ts = T_T0 * T0

    return Ps, Ts, Ms

if __name__ == "__main__":
    P4 = 200000
    T4 = 2000
    M4 = 0.3
    m_dot = 5.0
    length = 0.1
    depth = 1.0

    # Converging section (4 -> 5)
    P_conv, T_conv, M_conv, m_dot, A_conv, h_conv, x_conv = design_converging_section(P4, T4, M4, m_dot, length, depth)

    P5 = P_conv[-1]
    T5 = T_conv[-1]
    M5 = M_conv[-1]
    A5 = A_conv[-1]
    h5 = h_conv[-1]

    print("Section 5")
    print(f"P5 = {P5} Pa")
    print(f"T5 = {T5} K")
    print(f"M5 = {M5}")
    print(f"mdot = {m_dot} kg/s")
    print(f"A5 = {A5} m^2")
    print(f"h5 = {h5} m")

    # Nozzle section (5 -> 6)
    P_exit = 9112.32

    P_nozzle, T_nozzle, M_nozzle, m_dot, A_nozzle, h_nozzle, x_nozzle, c_plus_nozzle, c_minus_nozzle = design_nozzle(P5, T5, M5, m_dot, P_exit, depth)

    P6 = P_nozzle[-1]
    T6 = T_nozzle[-1]
    M6 = M_nozzle[-1]
    A6 = A_nozzle[-1]
    h6 = h_nozzle[-1]

    print("Section 6")
    print(f"P6 = {P6} Pa")
    print(f"T6 = {T6} K")
    print(f"M6 = {M6}")
    print(f"mdot = {m_dot} kg/s")
    print(f"A6 = {A6} m^2")
    print(f"h6 = {h6} m")

    # Nozzle Plots
    # Geometry
    plt.figure()
    plt.plot(x_nozzle, np.zeros(len(x_nozzle)), 'r-', linewidth=2)
    plt.plot(x_nozzle, -h_nozzle, 'r-', linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('h (m)')
    plt.title('Nozzle Geometry')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # Area
    plt.figure()
    plt.plot(x_nozzle, A_nozzle, linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('A (m^2)')
    plt.title('Nozzle Area Distribution')
    plt.grid(True)
    plt.show()

    # Pressure
    plt.figure()
    plt.plot(x_nozzle, P_nozzle, linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('P (Pa)')
    plt.title('Nozzle Pressure Distribution')
    plt.grid(True)
    plt.show()

    # Temperature
    plt.figure()
    plt.plot(x_nozzle, T_nozzle, linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('T (K)')
    plt.title('Nozzle Temperature Distribution')
    plt.grid(True)
    plt.show()

    # Mach number
    plt.figure()
    plt.plot(x_nozzle, M_nozzle, linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('M')
    plt.title('Nozzle Mach Number Distribution')
    plt.grid(True)
    plt.show()

    # Combined contour
    x_nozzle_offset = x_nozzle + x_conv[-1]  # Offset nozzle to start at end of converging section
    plt.figure(figsize=(12, 6))
    plt.plot(x_conv, -h_conv, 'b-', linewidth=2, label='Converging Section')
    plt.plot(x_conv, np.zeros(len(x_conv)), 'b-', linewidth=2)
    plt.plot(x_nozzle_offset, -h_nozzle, 'r-', linewidth=2, label='Nozzle')
    plt.plot(x_nozzle_offset, np.zeros(len(x_nozzle)), 'r-', linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('r (m)')
    plt.title('Complete Ramjet Nozzle Geometry')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # MOC Analysis
    P5 = 90000.82689383549
    T5 = 1300
    M5 = 1

    Ps, Ts, Ms, c_plus_chars, c_minus_chars = analyze_nozzle_moc(x_nozzle, h_nozzle, P5, T5, M5, depth)

    print("On-Design MOC Analysis")
    print(f"P6 = {Ps[-1]} Pa")
    print(f"T6 = {Ts[-1]} K")
    print(f"M6 = {Ms[-1]}")

    # Plot characteristics and nozzle geometry
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot C- characteristics (red)
    for c_minus in c_minus_chars:
        path = np.array(c_minus['path'])
        ax.plot(path[:, 0], path[:, 1], 'r-', alpha=0.3, linewidth=0.5)
        ax.plot(path[:, 0], -path[:, 1], 'r-', alpha=0.3, linewidth=0.5)
    
    # Plot C+ characteristics (blue)
    for c_plus in c_plus_chars:
        path = np.array(c_plus['path'])
        ax.plot(path[:, 0], path[:, 1], 'b-', alpha=0.3, linewidth=0.5)
        ax.plot(path[:, 0], -path[:, 1], 'b-', alpha=0.3, linewidth=0.5)
    
    # Plot nozzle wall (green)
    ax.plot(x_nozzle, h_nozzle, 'g-', linewidth=3, label='Nozzle Wall')
    ax.plot(x_nozzle, -h_nozzle, 'g-', linewidth=3)
    
    # Plot centerline
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Centerline')
    
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title('MOC Characteristic Network and Nozzle Geometry', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(x_nozzle, Ps, linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('P (Pa)')
    plt.title('On-Design MOC Analysis: Pressure')
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(x_nozzle, Ts, linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('T (K)')
    plt.title('On-Design MOC Analysis: Temperature')
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(x_nozzle, Ms, linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('M')
    plt.title('On-Design MOC Analysis: Mach Number')
    plt.grid(True)
    plt.show()