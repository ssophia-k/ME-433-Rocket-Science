import numpy as np

def generate_parabola_profile(a, num_points=100):
    """
    Generate a parabolic front-end profile. 
    Returns:
        dict with 'x', 'y_top', and 'y_bot' arrays.
    """
    x = np.linspace(0, 1, num_points)
    y_top = a * np.sqrt(x)+0.5
    y_bottom = -a * np.sqrt(x)+0.5
    y_top = np.clip(y_top, 0.0, 1.0)
    y_bottom = np.clip(y_bottom, 0.0, 1.0)
    # else:
    #     x_max = (1/(2*a))**2
    #     x = np.linspace(max(0, 1 - x_max), 1, num_points)
    #     y_top = a * np.sqrt(x+x_max-1)+0.5
    #     y_bottom = -a * np.sqrt(x+x_max-1)+0.5
    return {
        'x': x,
        'y_top': y_top,
        'y_bot': y_bottom,
        'a_param': a,
        'num_points': num_points,
    }


def generate_triangle_profile(half_angle):
    """
    Generate a minimal triangular profile. Automatically determines the
    minimal number of x-points needed to describe the profile.

    If the triangle is unclipped, only two x-points are used.
    If the angle causes clipping (triangle + rectangle), three x-points are used.

    Returns:
        dict with 'x', 'y_top', and 'y_bot' arrays.
    """
    # height from 0 to 1, centered at 0.5
    # full triangle spans x in [0, 1] unless clipped
    # if left_start:
        # compute where y_top hits 1 or y_bottom hits 0
    slope = np.tan(np.deg2rad(half_angle))
    x_clip = (1 - 0.5) / slope if slope > 0 else np.inf
    # x_bot_clip = (0 - 0.5) / (-slope) if slope > 0 else np.inf
    # x_clip = min(x_top_clip, x_bot_clip)

    if x_clip >= 1:
        # unclipped triangle
        x = np.array([0, 1])
    else:
        # triangle + rectangle region
        x = np.array([0, x_clip, 1])

    y_top = np.clip(slope * (x) + 0.5, 0, 1)
    y_bottom = np.clip(-slope * (x) + 0.5, 0, 1)

    return {
        'x': x,
        'y_top': y_top,
        'y_bot': y_bottom,
        'half_angle': half_angle
    }

def generate_triangular_wedge_profile(half_angle, base_height):
    top_bound = 0.5 + base_height/2
    bot_bound = 0.5 - base_height/2
    slope = np.tan(np.deg2rad(half_angle))
    x_clip = (top_bound - 0.5) / slope if slope > 0 else np.inf

    if x_clip >= 1:
        # unclipped triangle, base height also doesn't rly matter
        x = np.array([0, 1])
        y_top = np.array([0.5, slope+0.5])
        y_bot = np.array([0.5, 0.5-slope])
    else:
        # triangle + rectangle region
        x = np.array([0, x_clip, 1])
        y_top = np.array([0.5, top_bound, top_bound])
        y_bot = np.array([0.5, bot_bound, bot_bound])

    return {
        'x': x,
        'y_top': y_top,
        'y_bot': y_bot,
        'half_angle': half_angle,
        'base_height': base_height
    }

def generate_trapezoid_profile(height, top_length):
    """
    Generate a minimal triangular profile. Automatically determines the
    minimal number of x-points needed to describe the profile.

    If the triangle is unclipped, only two x-points are used.
    If the angle causes clipping (triangle + rectangle), three x-points are used.

    Returns:
        dict with 'x', 'y_top', and 'y_bot' arrays.
    """
    x = np.array([0, (1 - top_length), 1.0])
    y_top = np.array([0, height, height])
    y_bottom = np.array([0, 0, 0])

    return {
        'x': x,
        'y_top': y_top,
        'y_bot': y_bottom,
        'height': height,
        'top_length': top_length
    }

def generate_power_series_profile(base_height, num_points=100):
    """
    
    Returns:
        dict with 'x', 'y_top', and 'y_bot' arrays.
    """
    x = np.linspace(0, 1, num_points)
    y_top = base_height * x**0.66 +0.5
    y_bottom = -base_height * x**0.66 +0.5
    y_top = np.clip(y_top, 0.0, 1.0)
    y_bottom = np.clip(y_bottom, 0.0, 1.0)
    # else:
    #     x_max = (1/(2*a))**2
    #     x = np.linspace(max(0, 1 - x_max), 1, num_points)
    #     y_top = a * np.sqrt(x+x_max-1)+0.5
    #     y_bottom = -a * np.sqrt(x+x_max-1)+0.5
    return {
        'x': x,
        'y_top': y_top,
        'y_bot': y_bottom,
        'base_height': base_height,
        'num_points': num_points,
    }

def generate_diamond_ish_profile(base_height, slope):
    h = base_height
    m = slope
    A = (1+h)/2
    B = (1-h)/2
    intersect_x = (A-0.5+m)/(2*m)
    intersect_y = m*intersect_x + 0.5
    if(intersect_y <=1):
        x = np.array([0, intersect_x, 1])
        y_top = np.array([0.5, intersect_y, A])
        y_bot = np.array([0.5, 1-intersect_y, B])
    else: #clipping
        clip_x_1 = 0.5/m
        clip_x_2 = (A-1)/m + 1
        x = np.array([0, clip_x_1, clip_x_2, 1])
        y_top = np.array([0.5, 1, 1, A])
        y_bot = np.array([0.5, 0, 0, B])
    
    return {
        'x': x,
        'y_top': y_top,
        'y_bot': y_bot,
        'base_height': base_height,
        'slope': slope,
    }
