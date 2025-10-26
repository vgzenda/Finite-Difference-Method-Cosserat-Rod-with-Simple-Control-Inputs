import numpy as np
import matplotlib.pyplot as plt
from cosserat_rod import Cosserat_Rod
from visualize import *


# # initialize cosserat rod (controls)
length = 1.0
num_nodes = 30  # Number of spatial points
rod = Cosserat_Rod(length=length, num_nodes=num_nodes)

# Update BDF coefficients after setting dt
rod.dt = 0.025  # Time step
rod.N_steps = 500  # Number of time steps

# Initial conditions
target_Lambda_tip = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
g_SE0 = np.eye(4)
mcV0 = np.zeros((6,))


#------------------ Constant tension forces --------------------
# Define controls: Constant tendon tensions
# Apply tension to tendon 0 (e.g., in +x direction at 45° from center)
# rod.tendon_tensions = np.array([5.0, 0.0, 0.0, 0.0])  # 5N on tendon 0
# tendon_forces = np.dot(rod.tendon_tensions, rod.tendon_dirs)  # Shape: (3,)

# Time integration with constant tendon forces
# g_SE_list, xi_list, eta_list, Lambda_list, Lambda0_list = rod.time_integrate_cosserat_rod_over_controls(
#     g_SE0, eta0, target_Lambda_tip, tendon_forces=tendon_forces, verbose=False)

#------------------ Constant tension forces --------------------

# Define controls: Time-varying tendon tensions (sinusoidal)
def tendon_control(t):
    # Sinusoidal tension on tendon 0 and 2 (opposing directions)
    freq = 1.0  # Hz
    amplitude = 1.0  # Max 5N
    tensions = np.zeros(4)
    tensions[0] = amplitude * np.sin(2 * np.pi * freq * t)  # Tendon 0
    tensions[2] = amplitude * np.sin(2 * np.pi * freq * t + np.pi)  # Tendon 2 (opposite phase)

    tensions[1] = amplitude * np.cos(2 * np.pi * freq * t)  # Tendon 0
    tensions[3] = amplitude * np.cos(2 * np.pi * freq * t + np.pi)  # Tendon 2 (opposite phase)
    return np.dot(tensions, rod.tendon_dirs)  # Convert to force vector

# Time integration with constant tendon forces
g_SE_list, mcE_list, mcV_list, Lambda_list, Lambda0_list = rod.time_integrate_cosserat_rod_over_controls(
    g_SE0, mcV0, target_Lambda_tip, tendon_forces=tendon_control, verbose=False)


# ---- Animate results
fig, anim = rod.animate_cosserat_rod(g_SE_list, mcE_list, mcV_list, Lambda_list)
plt.show()

# Analyze dynamics (optional, for additional insights)
fig = rod.analyze_rod_dynamics(g_SE_list, mcE_list, mcV_list)
# anim.save('rod_animation.mp4', writer='ffmpeg', fps=30, dpi=150)

plt.show()
