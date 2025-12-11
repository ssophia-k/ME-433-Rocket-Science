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


def T2_T1_from_normal_shock(M_in, gamma):
    """
    Get out / in temperature ratio for N.S
    Parameters:
        incoming Mach (unitless)
        gamma (unitless)
    Returns:
        T2/T1 (unitless)
    """
    Pr = 1+2*gamma/(gamma+1)*(M_in**2-1)
    rhor = (gamma+1)*M_in**2/(2+(gamma-1)*M_in**2)
    Tr=Pr/rhor
    return Tr
    

def rho2_rho1_from_normal_shock(M_in, gamma):
    """
    Get out / in density ratio for N.S
    Parameters:
        incoming Mach (unitless)
        gamma (unitless)
    Returns:
        rho2/rho1 (unitless)
    """
    rhor = (gamma+1)*M_in**2/(2+(gamma-1)*M_in**2)
    return rhor
