import numpy as np

def generate_parabola_profile(a, num_points=100):
    """
    Generate a parabolic front-end profile. Left/right start will determine
    if we construct from left to right or right to left (so if we clip off any portion of the parabola, or if it simply remains smaller).
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
    x_top_clip = (1 - 0.5) / slope if slope > 0 else np.inf
    x_bot_clip = (0 - 0.5) / (-slope) if slope > 0 else np.inf
    x_clip = min(x_top_clip, x_bot_clip)

    if x_clip >= 1:
        # unclipped triangle
        x = np.array([0, 1])
    else:
        # triangle + rectangle region
        x = np.array([0, x_clip, 1])

    y_top = np.clip(slope * x + 0.5, 0, 1)
    y_bottom = np.clip(-slope * x + 0.5, 0, 1)

    return {
        'x': x,
        'y_top': y_top,
        'y_bot': y_bottom,
        'half_angle': half_angle
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

