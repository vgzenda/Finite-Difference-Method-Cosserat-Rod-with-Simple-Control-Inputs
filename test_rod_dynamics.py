import numpy as np
import matplotlib.pyplot as plt
from cosserat_rod import Cosserat_Rod
from visualize import *


# ------- Test the spatial integration of rod dynamics
# if __name__ == "__main__":
#     # initialize cosserat rod
#     length = 1.0
#     num_steps = 20 # Number of spatial points
#     rod = Cosserat_Rod(length=length,num_steps=num_steps)
#     N = num_steps # Number of spatial points
#     x, w = np.polynomial.legendre.leggauss(N)
#     print('x',x[-1])
#     s_points = np.linspace(0, length, N)  # arc length parameter
#     # s_points = x/2 + np.ones_like(x)/2
#     rod.dt = 0.025
#     # parameters
#     mbM, mbK, mbD = rod.soft_robot_parameters() 
#     # Initial conditions
#     g_SE0 = np.eye(4)  # Initial configuration in SE(3)
#     F_ext0 = rod.gravity_wrench(g_SE0,mbM)
#     xi0 = np.linalg.inv(mbK) @ (-F_ext0) + rod.xi_rest(0.0)  # Initial strain field
#     eta0 = np.zeros(6)  # Initial velocity field
#     xi_star = rod.xi_rest(0.0)  # Rest strain field
#     Lambda0 = mbK@(xi0 - xi_star)
#     xi_prev = np.array([xi_star] * N)    # Previous strain field (for semi-discretization)
#     xi_pprev = np.array([xi_star] * N)   # Previous strain field (for semi-discretization)
#     eta_prev = np.zeros((N,6))           # Previous strain field (for semi-discretization)
#     eta_pprev = np.zeros((N,6))          # Previous strain field (for semi-discretization)
    
#     tendon_forces = np.dot(rod.tendon_tensions, rod.tendon_dirs)

#     # Integrate the rod
#     g_SE, xi, eta, Lambda,  Lambda_prime, eta_prime = rod.integrate_rod(Lambda0, g_SE0, eta0, xi_prev, xi_pprev, eta_prev,eta_pprev, s_points,tendon_forces)
    
#     visualize_rod(g_SE,eta=eta,xi=xi)
#     # Print results
#     print("Final configuration in SE(3):")
#     print(g_SE[-1])
#     print("Final strain field:")
#     print(xi[-1])
#     print("Final velocity field:")
#     print(eta[-1])



# # # initialize cosserat rod (no controls)
length = 1.0
num_nodes= 30 # Number of spatial points
rod = Cosserat_Rod(length=length,num_nodes=num_nodes)

target_Lambda_tip = np.array([0.0,0.0,0.0,0.0,1.3,0.0])
g_SE0 = np.eye(4)
eta0 = np.zeros((6,))

rod.dt = 0.025 # time step
rod.N_steps=200 # number of time steps

g_SE_list, xi_list, eta_list, Lambda_list, Lambda0_list = rod.time_integrate_cosserat_rod(g_SE0,eta0,target_Lambda_tip,verbose=False) 

fig, anim = rod.animate_cosserat_rod(g_SE_list, xi_list, eta_list, Lambda_list)
plt.show()