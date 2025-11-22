import numpy as np

def point_in_triangle(p, a, b, c):
    """
    p, a, b, c are (x, y) tuples
    Returns True if p is inside triangle abc, else False.
    """

    (px, py) = p
    (ax, ay) = a
    (bx, by) = b
    (cx, cy) = c

    # Vectors
    v0 = (cx - ax, cy - ay)
    v1 = (bx - ax, by - ay)
    v2 = (px - ax, py - ay)

    # Dot products
    dot00 = v0[0]*v0[0] + v0[1]*v0[1]
    dot01 = v0[0]*v1[0] + v0[1]*v1[1]
    dot02 = v0[0]*v2[0] + v0[1]*v2[1]
    dot11 = v1[0]*v1[0] + v1[1]*v1[1]
    dot12 = v1[0]*v2[0] + v1[1]*v2[1]

    # Compute barycentric coordinates
    denom = dot00 * dot11 - dot01 * dot01
    if denom == 0:
        return False  # Degenerate triangle

    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom

    # Inside check
    return (u >= 0 - 1e-12) and (v >= 0 - 1e-12) and (u + v <= 1 + 1e-12)


