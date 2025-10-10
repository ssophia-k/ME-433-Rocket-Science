import numpy as np
import matplotlib.pyplot as plt

# Givens
M_in   = 7.0
P_in   = 101325.0 # Pa
T_in   = 300.0 # K
depth  = 1.0 # m

R_air     = 287.052874
gamma_air = 1.4
a_in      = np.sqrt(gamma_air * R_air * T_in)

# exit-plane data (rows: y, p, rho, u)
data = np.loadtxt("HW2/HW2_Tunnel.txt", delimiter=",")
y, pressure, density, velocity = data

# Sort by y to be safe (but i checked and it was sorted already)
idx = np.argsort(y)
y, pressure, density, velocity = [arr[idx] for arr in (y, pressure, density, velocity)]

# Quick plots
plt.figure(); plt.plot(pressure, y); plt.ylabel("y [m]"); plt.xlabel("Pressure [Pa]")
plt.title("Pressure Profile at Exit"); plt.grid(); plt.savefig("HW2/pressure_profile.png"); plt.close()

plt.figure(); plt.plot(density, y); plt.ylabel("y [m]"); plt.xlabel("Density [kg/m^3]")
plt.title("Density Profile at Exit"); plt.grid(); plt.savefig("HW2/density_profile.png"); plt.close()

plt.figure(); plt.plot(velocity, y); plt.ylabel("y [m]"); plt.xlabel("Velocity [m/s]")
plt.title("Velocity Profile at Exit"); plt.grid(); plt.savefig("HW2/velocity_profile.png"); plt.close()

# Inlet (uniform) properties
rho_in = P_in / (R_air * T_in)
u_in = M_in * a_in

span_y = np.ptp(y) # y_max - y_min
A_in = depth * span_y

# Momentum-flux integrals: rho * u^2 dA
momentum_in  = rho_in * u_in**2 * A_in
momentum_out = depth * np.trapz(density * velocity**2, y)

# Pressure integrals : p dA
P_in_int  = P_in * A_in
P_out_int = depth * np.trapz(pressure, y)

# Drag force is the sum of these (outward normal sign convention)
Drag = (momentum_out - momentum_in) + (P_out_int - P_in_int)
print(f"Drag: {Drag:.6e} N") # drag will be negative since it points to the left (and +x is to the right for us)

# Exit temperature via ideal gas
temperature = pressure / (density * R_air)
plt.figure(); plt.plot(temperature, y)
plt.ylabel("y [m]"); plt.xlabel("Ideal-Gas Temperature [K]")
plt.title("Temperature Profile at Exit"); plt.grid(); plt.show()
