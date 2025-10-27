import numpy as np

def M2_from_normal_shock(M_in, gamma):
    """
    Normal shock solution for M2 given M1
    Parameters:
        incoming Mach (unitless)
        gamma (unitless)
    Returns:
        outgoing Mach (unitless)
    """
    return np.sqrt( (1 + (gamma-1)/2 * M_in**2) / (gamma * M_in**2 - (gamma-1)/2))


def P2_P1_from_normal_shock(M_in, gamma):
    """
    Get out / in pressure ratio for N.S
    Parameters:
        incoming Mach (unitless)
        gamma (unitless)
    Returns:
        P2/P1 (unitless)
    """
    return 1 + ( (2 * gamma) / (gamma + 1)) * (M_in**2 - 1)