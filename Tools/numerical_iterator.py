import numpy as np

def numerical_iterator(func, start, end, goal_y, tol=1e-6, max_iter=1000):
    """
    Find x in [start, end] such that func(x) ~= goal_y using the bisection method.
    Uses the bisection method to find the x value that gives the desired y value for a given function
    Want to find x value in [start, end] (inclusive) such that func(x) = goal_y
    ASSUMES THAT FUNCTION IS CONTINUOUS OVER INTERVAL AND MONOTONICITY FOR IT TO WORK PROPERLY
    Stops when |func(x) - goal_y| <= tol, or (optionally) when the interval is <= xtol

    func - function to evaluate
    start - starting x value
    end - ending x value
    goal_y - desired y value
    tol - num tolerance for convergence
    max_iter - maximum number of iterations
    """

    if start > end:
        # The search interval must be valid (go left to right)
        raise ValueError("start must be <= end.")

    # We can define g(x) = func(x) - goal_y to turn this into a root-finding problem
    # Want to find g(x) = 0
    # This makes it easier because picking the correct half of the interval will just rely on the sign of g
    g_start = func(start) - goal_y  # Evaluate at left endpoint
    # If the left endpoint already satisfies the tolerance, return it immediately
    if abs(g_start) <= tol:
        return start
    g_end = func(end) - goal_y      # Evaluate at right endpoint
    # If the right endpoint already satisfies the tolerance, return it immediately
    if abs(g_end) <= tol:
        return end

    # If the interval is simply a point AND that point is not the solution, then you messed up...
    if start == end:
        raise ValueError("start == end but goal not found.")

    # For bisection to work, [f(start), f(end)] must include the target value
    # AKA g_start and g_end must have opposite signs 
    # THIS ASSUMES MONOTONICITY
    if g_start * g_end > 0:
        raise ValueError("goal_y must be included in your interval.")

    # Start the bisection iteration
    for _ in range(max_iter):
        # Midpoint of the current interval
        mid = (start + end) / 2.0
        # Evaluate g at the midpoint
        g_mid = func(mid) - goal_y

        # If the midpoint is good enough, return it
        if abs(g_mid) <= tol:
            return mid

        # Otherwise, continue the bisection
        # Decide which half of the interval still brackets the root (which means it has g values of opposite signs)
        # Recall that we turned this into a root-finding problem
        # Whatever half contains the root will have g values of opposite signs on the endpoints
        if g_start * g_mid <= 0:
            # Keep the left half: update right endpoint and its function value.
            end, g_end = mid, g_mid
        else:
            # Keep the right half: update left endpoint and its function value.
            start, g_start = mid, g_mid


    # If we exit the loop, we didn't reach the tolerance within max_iter iterations.
    raise RuntimeError("Maximum number of iterations reached without convergence.")
