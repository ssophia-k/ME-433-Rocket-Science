import numpy as np
from matplotlib import pyplot as plt
import os, sys
from pathlib import Path

# --- Path setup for Tools and HW4 imports ---
sys.path.insert(0, os.fspath(Path(__file__).parents[1]))
from Tools.oblique_shock import *
from Tools.expansion_fan import *
from Tools.isentropic import P0_P
from Tools.misc_functions import get_speed_of_sound
from Tools.constants import *
from HW4.P3_helpers import get_surface_angles, get_turn_angles


# --------------- Configuration ---------------
M0 = 3
gamma = 1.4
P0r = P0_P(M0, gamma)
alphas = np.linspace(-5, 5, 1000)  # degrees


# --------------- Geometry Loader ---------------
def load_airfoil_geometry(airfoil_id):
    """Load x, y coordinates for a given airfoil file."""
    with open(f"HW4/airfoils/airfoil{airfoil_id}.txt", "r") as file:
        xs = [float(i) for i in file.readline().strip().split(",")]
        ys = [float(i) for i in file.readline().strip().split(",")]
    return xs, ys


# --------------- Flow Physics Core ---------------
def compute_pressure_and_mach(M_in, P_in, turn_deg, gamma):
    """
    Handles oblique shock (positive turn) and expansion wave (negative turn) logic.
    turn_deg > 0 => oblique shock
    turn_deg < 0 => expansion wave
    """
    if turn_deg > 0:
        # Oblique shock
        _, Pr, _, M2 = mach_function(M_in, gamma, turn_deg)
        P_out = P_in * Pr
    elif turn_deg < 0:
        # Expansion fan
        M2 = get_M2_from_nu(M_in, gamma, -turn_deg)
        p0_over_p_in = P0_P(M_in, gamma)
        p0_over_p_out = P0_P(M2, gamma)
        P_out = P_in * (p0_over_p_in / p0_over_p_out)
    else:
        # No flow deflection
        M2 = M_in
        P_out = P_in
    return M2, P_out


# --------------- Aerodynamic Force Calculation ---------------
def compute_panel_forces(xs, ys, alpha_rad, turn_angles_top, turn_angles_bot):
    """
    Compute total lift and drag forces by marching along each surface.
    Uses precomputed turn angles (deg) for top and bottom surfaces.
    """
    P_top, P_bottom, M_top, M_bottom = [], [], [], []
    total_lift = 0.0
    total_drag = 0.0
    Npan = len(xs) - 1

    for i in range(Npan):
        dx = xs[i+1] - xs[i]
        dy = ys[i+1] - ys[i]
        panel_theta = np.arctan2(dy, dx)

        turn_top_deg = turn_angles_top[i]
        turn_bot_deg = turn_angles_bot[i]

        # --- Top surface ---
        M_in_top = M0 if i == 0 else M_top[-1]
        P_in_top = 1.0 if i == 0 else P_top[-1]
        M2_top, P_out_top = compute_pressure_and_mach(M_in_top, P_in_top, turn_top_deg, gamma)
        M_top.append(M2_top)
        P_top.append(P_out_top)

        # --- Bottom surface ---
        M_in_bot = M0 if i == 0 else M_bottom[-1]
        P_in_bot = 1.0 if i == 0 else P_bottom[-1]
        M2_bot, P_out_bot = compute_pressure_and_mach(M_in_bot, P_in_bot, turn_bot_deg, gamma)
        M_bottom.append(M2_bot)
        P_bottom.append(P_out_bot)

        # --- Pressure difference and aerodynamic forces ---
        dp = P_out_bot - P_out_top
        section_length = np.hypot(dx, dy)

        # Panel normal (pointing outward from top surface)
        nx = -np.sin(panel_theta)
        ny =  np.cos(panel_theta)

        Fx = dp * section_length * nx
        Fy = dp * section_length * ny

        # Resolve into lift and drag relative to freestream
        drag_panel = Fx * np.cos(alpha_rad) + Fy * np.sin(alpha_rad)
        lift_panel = -Fx * np.sin(alpha_rad) + Fy * np.cos(alpha_rad)

        total_drag += drag_panel
        total_lift += lift_panel

    return total_lift, total_drag


# --------------- Analysis Loop ---------------
def analyze_airfoil(xs, ys):
    """Compute lift and drag across angles of attack for one airfoil."""
    lifts = []
    drags = []
    Npan = len(xs) - 1

    for alpha_deg in alphas:
        alpha_rad = np.deg2rad(alpha_deg)

        # Use provided helper functions to compute turning angles for top/bottom
        turn_angles_top = get_turn_angles(xs, ys, alpha_deg, top=True)
        turn_angles_bot = get_turn_angles(xs, ys, alpha_deg, top=False)

        # Ensure number of turns matches number of panels
        turn_angles_top = turn_angles_top[:Npan]
        turn_angles_bot = turn_angles_bot[:Npan]

        lift, drag = compute_panel_forces(xs, ys, alpha_rad, turn_angles_top, turn_angles_bot)
        lifts.append(lift)
        drags.append(drag)

    return drags, lifts


# --------------- Main Execution ---------------
def main():
    for airfoil_id in range(1, 5):  # Adjust as needed
        xs, ys = load_airfoil_geometry(airfoil_id)
        drags, lifts = analyze_airfoil(xs, ys)
        plt.plot(drags, lifts, label=f"airfoil {airfoil_id}")

    plt.xlabel("C_d")
    plt.ylabel("C_l")
    plt.title("C_l vs C_d for alpha from -5 to 5 degrees")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
