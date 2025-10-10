# Acceleration due to gravity
g = 9.81  # m/s^2

# Specific gas constant for dry air
R_air = 287.052874 # J/(kg*K)

# Specific heat capacity at constant pressure for dry air
C_p_air = 1005 # J/(kg*K)
C_v_air = C_p_air - R_air # J/(kg*K)
gamma_air = 1.4 # dimensionless

# Reference pressures and temperatures
P_sea = 101325 # Pa
T_sea = 288.15 # K

# Temperature lapse rates in the atmosphere
k_0_11 = -6.5 # K/km, gradient from 0 to 11 km
k_11_20 = 0.0 # K/km, gradient from 11 to
k_20_32 = 1.0 # K/km, gradient from 20 to 32 km
k_32_47 = 2.8 # K/km, gradient from 32 to 47 km

# Altitudes defining the layers
altitude_bounds = [0, 11, 20, 32, 47] # km

#Boltzmann constant
k_b = 1.380649e-23 # J/K

# Molecular diameters (in meters)
diameter_N2 = 3.64 * 10**(-10) # m