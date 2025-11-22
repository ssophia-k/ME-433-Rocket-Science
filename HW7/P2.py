"""
Once you have designed your nozzle it will have some profile of Area as a function of
distance. Using this area profile construct the theoretical distribution of Mach number,
Pressure, and Temperature that quasi 1-D flow would predict.
"""
import numpy as np
import matplotlib.pyplot as plt

import os, sys
from pathlib import Path
sys.path.insert(0, os.fspath(Path(__file__).parents[1]))

from Tools.constants import *
from HW7.helpers import *  
from HW5.helpers import *
from Tools.isentropic import *
from Tools.expansion_fan import *

# Givens
G = 1.4 # gamma of air
M_in = 1 # incoming Mach number
P_in = 101325 # Pa, incoming pressure
T_in = 300 # K, incoming temperature
A_in = 1 # m, height of nozzle inflow
D = 1 # m, constant depth of nozzle into page
M_exit = 3 # exit Mach number after the straightening section


# From Problem 1:
x_nozzle_wall = np.array([
    0.0, 0.7346686821945279, 0.8589484606639747, 0.9583541594850826,
    1.0478595859692252, 1.132425428559468, 1.2144626602425355,
    1.295368207471912, 1.376048321460895, 1.457141952891978,
    1.539130212310524, 1.6223955362499531, 1.7072561206416172,
    1.7939871651011285, 1.882834622377215, 1.9740244658467054,
    2.0677691624513614, 2.164272340639305, 2.2637322588642355,
    2.3663444570858325, 2.4723038402849937, 2.5818063613948374,
    2.6950504177478645, 2.8122380413256955, 2.9335759407321444,
    3.059276435894461, 3.1895583183251075, 3.324647659207069,
    3.464778584827512, 3.6101940339451875, 3.7611465087559672,
    3.9178988297722617, 4.080724903215015, 4.249910508311924,
    4.4257541121082555, 4.608567717064849, 4.798677749090501,
    4.996425991322502, 5.202170569398788, 5.416286995848397,
    5.639169279108302, 5.871231103481928, 6.112907088863967,
    6.364654136919891, 6.626952872140362, 6.90030918611325,
    7.185255896614685, 7.482354529785946, 7.792197238873979,
    8.111358354476398
])

y_nozzle_wall = np.array([
    0.5, 0.8157487729082695, 0.8680609128296178, 0.909022110027308,
    0.9451106380832339, 0.9784581469963785, 1.010081554152531,
    1.0405517773630806, 1.0702221503415659, 1.0993259825563118,
    1.1280243484328536, 1.156431867798111, 1.1846316459569386,
    1.2126844221422477, 1.2406344189237026, 1.2685132121778373,
    1.296342360841522, 1.3241352300268416, 1.351898272501665,
    1.3796319356727313, 1.4073313026077472, 1.434986539573187,
    1.462583199191022, 1.490102413301494, 1.5175209995818806,
    1.5448114987094357, 1.5719421546079748, 1.5988768461740417,
    1.625574976979621, 1.6519913273344684, 1.6780758716846949,
    1.7037735633798439, 1.7290240879333618, 1.7537615851929733,
    1.7779143404056346, 1.8014044433935545, 1.8241474150132073,
    1.8460517993332048, 1.867018719687729, 1.88694139657597,
    1.905704624724799, 1.9231842063593099, 1.9392463374954008,
    1.9537469433940604, 1.9665309589923476, 1.9774315496034163,
    1.9862692667545716, 1.9928511332406216, 1.9969696510291628,
    1.9969696510291628
])

# Original area (half-height * 2)
A_nozzle_wall = y_nozzle_wall * 2  # since this is half the nozzle height

# interpolate more values
N_fine = 400  # density
x_fine = np.linspace(x_nozzle_wall[0], x_nozzle_wall[-1], N_fine)

# linear interpolation of wall and area
y_fine = np.interp(x_fine, x_nozzle_wall, y_nozzle_wall)
A_fine = y_fine * 2.0

# Now we do what we did for HW 5
A_star = A_in # (since we know we start sonic)
x_star = 0.0  # (since we know we start sonic)

# Now find M at every other point (using the fine grid)
Mach_list = []
P_P0_list = []
T_T0_list = []

for x, area in zip(x_fine, A_fine):
    A_ratio_sq = (area / A_star)**2

    # if x <= x_star:
    #     # subsonic up to throat
    #     M = inverse_area_mach_relation(A_ratio_sq, G)[0]
    #     P_P0 = 1 / P0_P(M, G)
    # else:
    #     # supersonic
    M = inverse_area_mach_relation(A_ratio_sq, G)[1]
    P_P0 = 1 / P0_P(M, G)
    T_T0 = 1 / T0_T(M, G)  # T/T0 from isentropic relation

    Mach_list.append(M)
    P_P0_list.append(P_P0)
    T_T0_list.append(T_T0)

Mach_list = np.array(Mach_list)
P_P0_list = np.array(P_P0_list)
T_T0_list = np.array(T_T0_list)

# Stagnation from inlet
P0 = P_in * P0_P(M_in, G)   
T0 = T_in * T0_T(M_in, G)

P_list = P_P0_list * P0
T_list = T_T0_list * T0

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Mach number
axes[0].plot(x_fine, Mach_list)
axes[0].set_ylabel('Mach number, M')
axes[0].set_title('Quasi-1D Flow Along Nozzle Centerline')
axes[0].grid(True)

# Pressure 
axes[1].plot(x_fine, P_list)
axes[1].set_ylabel('Pressure [Pa]')
axes[1].grid(True)

# Temperature
axes[2].plot(x_fine, T_list)
axes[2].set_xlabel('x location [m]')
axes[2].set_ylabel('Temperature [K]')
axes[2].grid(True)

plt.tight_layout()
plt.show()

# Save Mach number, x, y matrix for later use in P3 (use fine grid)
np.save('HW7/data/x_nozzle_wall.npy', x_fine)
np.save('HW7/data/y_nozzle_wall.npy', y_fine)
np.save("HW7/data/P2_mach_list.npy", Mach_list)
np.save("HW7/data/P2_P_list.npy", P_list)
np.save("HW7/data/P2_T_list.npy", T_list)



