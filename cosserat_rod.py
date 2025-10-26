import numpy as np
import matplotlib.pyplot as plt 
from liegroups import * 
from visualize import * 
from scipy.optimize import minimize, least_squares

import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D



class Cosserat_Rod:
    def __init__(self,length,num_nodes):
        # Parameters 
        self.L = length                               # length
        self.N = num_nodes                            # number of elements (spatial discretization)
        self.s_points = np.linspace(0,self.L,self.N)
        print('self.s_points',self.s_points[-1])
        print('len(s_points)',len(self.s_points))
        self.dt = 0.0025                               # time step for BDF2
        # BDF2  coeffiencts
        self.c_0 = 1.5/self.dt
        self.c_1 = -2/self.dt 
        self.c_2 = 0.5/self.dt
        # num_time steps
        self.N_steps = 10
        # initial time
        self.t0 = 0.0
        

        # tendons
        self.n_tendons = 4                        # number of tendons
        self.tendon_tensions = np.zeros((4,))
        theta = np.pi/self.n_tendons              # tendon angle
        self.tendon_offset = 0.02                 # offset of the tendons from the center
        self.tendon_dirs = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [np.cos(theta + np.pi/2), np.sin(theta + np.pi/2), 0],
            [np.cos(theta + np.pi), np.sin(theta + np.pi), 0],
            [np.cos(theta + 3*np.pi/2), np.sin(theta + 3*np.pi/2), 0]]) # tendon directions

        # Boundary Conditions 
        self.g0 = np.eye(4)                       # initial config
        self.mcV0 = np.array([0, 0, 0, 0, 0, 0])  # initial velocity
        self.mcE0 = np.array([0, 0, 0, 0, 0, 0])  # initial strain
        self.target_Lambda_tip = np.zeros((6,))


    def update_bdf_coefficients(self):
        self.c_0 = 1.5 / self.dt
        self.c_1 = -2 / self.dt
        self.c_2 = 0.5 / self.dt



    def soft_robot_parameters(self):
        # Parameters from Samei's MATLAB code
        # F_0 = [0;0;1;0;0;0] # []         Free Static Strain under no Applied Loads (not used in these matrices)
        self.rho = 75e1          # [kg/m3]    Density of Material
        self.mu = 5e6           # [N/m^2s]   Viscosity of Peanut Butter
        self.r = 0.01             # [m]        Radius of Cross-Section
        self.E = 5e8             # [Pa]       Youngs Modulus
        G = self.E / (2 * (1 + 0.3))    # [Pa]       Shear Modulus
        A = np.pi * self.r**2          # [m2]       Cross-Sectional Area of Beam
        I = np.pi / 4 * self.r**4        # [m4]       2nd Moment of Inertia of Beam

        # Calculate J, Kbt, Kse, Cse, Cbt 
        # print('I:', I)
        J = np.diag([2 * I, I, I])            # [m4]       3D Moment of Inertia of Beam
        Kbt = np.diag([2 * G * I, self.E * I, self.E * I])    # [Nm^2]     Bending and Torsional Rigidity (Rotational)
        Kse = np.diag([self.E * A, G * A, G * A])      # [N]        Shear and Extension Rigidity (Linear)
        Cse = np.diag([3 * A, A, A]) * self.mu       # [N/s]      Shear and Extension Damping (Linear)
        Cbt = np.diag([2 * I, I, I]) * self.mu       # [N/s]      Bending and Torsional Damping (Rotational)

        m_linear = self.rho * A * np.eye(3)
        m_rotational = self.rho * J
        # print('det J', np.linalg.det(J))
        # print('det m_rotational', np.linalg.det(m_rotational))
        # print('det m_linear', np.linalg.det(m_linear))
        mbM = np.block([[m_rotational, np.zeros((3, 3))],
                        [np.zeros((3, 3)), m_linear]])


        mbK = np.block([[Kbt, np.zeros((3, 3))],
                        [np.zeros((3, 3)), Kse]])

        mbD = np.block([[Cbt, np.zeros((3, 3))],
                        [np.zeros((3, 3)), Cse]])
        # Print the determinants of the matrices
        # print('det MbM', np.linalg.det(mbM))
        # print('det mbK', np.linalg.det(mbK))
        # print('det mbD', np.linalg.det(mbD))

        return mbM, mbK, mbD


    # ================================================================== Equations of motion

    def actuation_wrench(self):
        Lambda_a = np.zeros((6,))
        return Lambda_a

    def gravity_wrench(self,g_SE,mbM):
        """Computes the gravity wrench F(s,t)."""
        G = np.array([0, 0, 0, 0, 0, -9.81])
        F_g = mbM@Adjoint_SE3(SE3_inv(g_SE))@G
        return F_g.reshape((6,))

    def mcE_rest(self,s):
        """Rest strain field mcE(s) for the Cosserat rod."""
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    #     return np.array([0.0, 0.0, 0.0,  np.sin(s), 0.0 , 0.0])


    def constitutive_law(self,s,mcE,Lambda_a,mbK,mbD,mcE_t):
        # rest strain 
        mcE_star = self.mcE_rest(s)
    #     print('Lambda_a',Lambda_a)
        # linear elastic constitucive law iwth active strain
        Lambda = mbK@(mcE - mcE_star) + Lambda_a + mbD@mcE_t
    #     print('Lambda',Lambda)
        return Lambda

    def get_strain(self, s, Lambda, Lambda_a, mbK, mbD, mcE_h):
        mcE_star = self.mcE_rest(s)
        mat = mbK + self.c_0 * mbD
        rhs = Lambda - Lambda_a + mbK @ mcE_star - mbD @ mcE_h
        mcE = np.linalg.solve(mat, rhs)
        return mcE


    def semi_discretized_cosserat_equations(self,s, mcE, mcV, mcE_h, mcV_h,mbM,mbK,mbD,F_ext,Lambda_a):
        """Computes the semi-discretized Cosserat rod equations."""

        mcV_t = self.c_0*mcV + mcV_h  # velocity field at time t
        mcE_t = self.c_0*mcE + mcE_h  # strain field at time t
        # velocity field
        mcV_prime = mcE_t + adjoint_SE3(mcV)@mcE
        # strain field
        Lambda = self.constitutive_law(s,mcE,Lambda_a,mbK,mbD,mcE_t)
        Lambda_prime = mbM@mcV_t - coadjoint_SE3(mcV)@mbM@mcV + coadjoint_SE3(mcE)@Lambda  - F_ext
        
        return Lambda_prime, mcV_prime


    def integrate_rod(self,Lambda0, g_SE0, mcV0, mcE_prev, mcE_pprev, mcV_prev,mcV_pprev, s_points,tendon_forces):
        """Integrates the semi-discretized Cosserat rod equations. in space"""
        # update the BDF coeffients
        self.update_bdf_coefficients()
        # allocate memory for the states
        Lambda = np.zeros((self.N, 6))  # stress field
        mcV = np.zeros((self.N, 6))  # velocity field
        g_SE = np.zeros((self.N, 4, 4))  # SE(3) configuration
        mcE = np.zeros((self.N, 6))  # strain field 
        Lambda_prime = np.zeros((self.N, 6))  # velocity field
        mcV_prime = np.zeros((self.N, 6))  # velocity field
        
        
        # set the history vectors from the previous time step
        mcE_h = self.c_1*mcE_prev + self.c_2*mcE_pprev
        mcV_h = self.c_1*mcV_prev + self.c_2*mcV_pprev

        # set initial conditions 
        Lambda[0] = Lambda0
        mcV[0] = mcV0
        g_SE[0] = g_SE0
        # set initial conditions for strain 
        mbM, mbK,mbD = self.soft_robot_parameters()
        # strain from stress
        mbK_inv = np.linalg.inv(mbK)
         # actuator
        Lambda_a = np.zeros((6,))
        mcE[0] = self.get_strain(s_points[0],Lambda[0],Lambda_a,mbK,mbD,mcE_h[0])
       

        # integration loop 
        for k in range(1,self.N):
            # print("Before")
            # visualize_frame(g_SE[k-1],mcV=mcV[k-1],mcE=mcE[k-1])
            s = s_points[k-1]  # current arc length parameter
            # print('s:',s)
            # step size
            ds = s_points[k] - s_points[k-1]  # step size in arc length
            # external loads 
            F_tendon = np.concatenate([np.zeros((3,)),tendon_forces])
            F_ext = self.gravity_wrench(g_SE[k-1],mbM) + F_tendon
            # Compute dynamics (Cosserat equations)
            Lambda_prime_vf, mcV_prime_vf = self.semi_discretized_cosserat_equations(s,mcE[k-1],mcV[k-1],mcE_h[k-1],mcV_h[k-1],mbM,mbK,mbD,F_ext,Lambda_a)
            # Euler spatial integration
            mcV[k] = mcV[k-1] + ds * mcV_prime_vf
            Lambda[k] = Lambda[k-1] + ds * Lambda_prime_vf
            # compute strain 
            # mcE[k] = self.get_strain(s,Lambda[k],Lambda_a,mbK)
            mcE[k] = self.get_strain(s,Lambda[k],Lambda_a,mbK,mbD,mcE_h[k])
            g_SE[k] = g_SE[k-1] @ exp_SE3(ds * mcE[k-1])
            # print('g_SE\n',g_SE[k])

            # print("After")
            # visualize_frame(g_SE[k],mcV=mcV[k],mcE=mcE[k])
            # print('p :', g_SE[k][:, 3])
            Lambda_prime[k] = Lambda_prime_vf
            mcV_prime[k] = mcV_prime_vf
        
        return g_SE, mcE, mcV, Lambda,  Lambda_prime, mcV_prime
        

    # ================================================================== Shooting Method

    def shooting_objective_function(self,Lambda0_guess, g_SE0, mcV0, mcE_prev, mcE_pprev, mcV_prev, mcV_pprev, 
                               s_points, target_Lambda_tip,tendon_forces):
        """
        Objective function for the shooting method.
        
        Args:
            Lambda0_guess: Initial guess for the stress field at s=0 (6,)
            g_SE0: Initial configuration in SE(3) (4,4)
            mcV0: Initial velocity field (6,)
            mcE_prev, mcE_pprev: Previous strain fields for semi-discretization (N,6)
            mcV_prev, mcV_pprev: Previous velocity fields for semi-discretization (N,6)
            s_points: Arc length parameter points (N,)
            target_Lambda_tip: Target stress field at the tip s=L (6,)
            N: Number of spatial points
            dt: Time step
        
        Returns:
            residual: Residual between integrated tip stress and target tip stress
        """
        try:
            # Integrate the rod with the guessed initial stress
            g_SE, mcE, mcV, Lambda, Lambda_prime, mcV_prime = self.integrate_rod(
                Lambda0_guess, g_SE0, mcV0, mcE_prev, mcE_pprev, mcV_prev, mcV_pprev, s_points,tendon_forces)
            
            # Get the stress at the tip (last point)
            Lambda_tip_integrated = Lambda[-1]
            
            # Compute residual
            residual = Lambda_tip_integrated - target_Lambda_tip
            
            # Return squared norm for scalar optimization or residual vector for least squares
            return residual
            
        except Exception as e:
            print(f"Error in shooting objective: {e}")
            # Return large residual if integration fails
            return np.ones(6) * 1e6

    def solve_bvp_shooting_method(self,g_SE0, mcV0, mcE_prev, mcE_pprev, mcV_prev, mcV_pprev, 
                                s_points, target_Lambda_tip,tendon_forces, Lambda0_initial_guess=None, method='least_squares', verbose=True):
        """
        Solve the Cosserat rod BVP using the shooting method.
        
        Args:
            g_SE0: Initial configuration in SE(3) (4,4)
            mcV0: Initial velocity field (6,)
            mcE_prev, mcE_pprev: Previous strain fields for semi-discretization (N,6)
            mcV_prev, mcV_pprev: Previous velocity fields for semi-discretization (N,6)
            s_points: Arc length parameter points (N,)
            target_Lambda_tip: Target stress field at the tip s=L (6,)
            Lambda0_initial_guess: Initial guess for Lambda0. If None, uses zero guess
            N: Number of spatial points
            dt: Time step
            method: Optimization method ('least_squares' or 'minimize')
            verbose: Print optimization progress
        
        Returns:
            result: Optimization result object
            solution: Dictionary containing the solution fields
        """
        
        # Initial guess for Lambda0 if not provided
        if Lambda0_initial_guess is None:
            # Use a reasonable initial guess based on the target tip stress
            # You might want to scale this based on your problem
            Lambda0_initial_guess = target_Lambda_tip * 0.1
        
        if verbose:
            print(f"Starting shooting method with initial guess: {Lambda0_initial_guess}")
            print(f"Target tip stress: {target_Lambda_tip}")
        
        # Define objective function for optimization
        def objective(Lambda0):
            return self.shooting_objective_function(
                Lambda0, g_SE0, mcV0, mcE_prev, mcE_pprev, mcV_prev, mcV_pprev,
                s_points, target_Lambda_tip,tendon_forces)
        
        # Solve using the specified method
        if method == 'least_squares':
            # Use scipy's least_squares for nonlinear least squares
            result = least_squares(
                objective, 
                Lambda0_initial_guess,
                method='lm',  # Levenberg-Marquardt
                verbose=2 if verbose else 0,
                ftol=1e-8,
                xtol=1e-8,
                max_nfev=1000
            )
            optimal_Lambda0 = result.x
            
        elif method == 'minimize':
            # Use scipy's minimize with scalar objective
            def scalar_objective(Lambda0):
                residual = objective(Lambda0)
                return np.sum(residual**2)
            
            result = minimize(
                scalar_objective,
                Lambda0_initial_guess,
                method='BFGS',
                options={'disp': verbose, 'maxiter': 1000}
            )
            optimal_Lambda0 = result.x
        
        else:
            raise ValueError("Method must be 'least_squares' or 'minimize'")
        
        # Integrate with the optimal initial stress to get the full solution
        g_SE, mcE, mcV, Lambda, Lambda_prime, mcV_prime = self.integrate_rod(
            optimal_Lambda0, g_SE0, mcV0, mcE_prev, mcE_pprev, mcV_prev, mcV_pprev,s_points,tendon_forces)
        
        # Package the solution
        solution = {
            'g_SE': g_SE,
            'mcE': mcE,
            'mcV': mcV,
            'Lambda': Lambda,
            'Lambda_prime': Lambda_prime,
            'mcV_prime': mcV_prime,
            'optimal_Lambda0': optimal_Lambda0,
            's_points': s_points
        }
        
        if verbose:
            print(f"Optimization completed. Success: {result.success}")
            if hasattr(result, 'cost'):
                print(f"Final residual norm: {np.sqrt(result.cost)}")
            print(f"Optimal Lambda0: {optimal_Lambda0}")
            print(f"Achieved tip stress: {Lambda[-1]}")
            print(f"Target tip stress: {target_Lambda_tip}")
            print(f"Tip stress error: {np.linalg.norm(Lambda[-1] - target_Lambda_tip)}")
        
        return result, solution

    def adaptive_shooting_method(self,g_SE0, mcV0, mcE_prev, mcE_pprev, mcV_prev, mcV_pprev,
                            s_points, target_Lambda_tip,tendon_forces,
                            max_attempts=5, scale_factors=None):
        """
        Adaptive shooting method that tries different initial guesses if the first attempt fails.
        
        Args:
            Same as solve_bvp_shooting_method, plus:
            max_attempts: Maximum number of attempts with different initial guesses
            scale_factors: List of scale factors to multiply target_Lambda_tip for initial guesses
        
        Returns:
            Same as solve_bvp_shooting_method
        """
        
        if scale_factors is None:
            scale_factors = [0.1, 0.5, 1.0, 2.0, 0.01]
        
        for attempt in range(max_attempts):
            try:
                scale = scale_factors[attempt % len(scale_factors)]
                Lambda0_guess = target_Lambda_tip * scale
                
                print(f"Attempt {attempt + 1}: Using scale factor {scale}")
                
                result, solution = self.solve_bvp_shooting_method(
                    g_SE0, mcV0, mcE_prev, mcE_pprev, mcV_prev, mcV_pprev,
                    s_points, target_Lambda_tip,tendon_forces, verbose=False)
                
                # Check if solution is reasonable
                tip_error = np.linalg.norm(solution['Lambda'][-1] - target_Lambda_tip)
                if tip_error < 1e-3 or result.success:
                    print(f"Converged on attempt {attempt + 1} with tip error: {tip_error}")
                    return result, solution
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                continue
        
        raise RuntimeError("Adaptive shooting method failed to converge after all attempts")


    # ================================================================== Time Integration

    def time_integrate_cosserat_rod(self,g_SE0,mcV0,target_Lambda_tip,verbose=False):
        # integrate the rod over time 
        tspan = [self.t0]
        t0 = self.t0
        # interval for rod 
        s_points = np.linspace(0, self.L, self.N)
        # Initial conditions
        mcE_star = self.mcE_rest(0.0)  # Rest strain field
        # Initial conditions 
        # mcE0 = np.linalg.inv(mbK) @ Lambda0 + mcE_star  # Initial strain field
        # mcE_star = mcE0
        mcE_prev = np.array([mcE_star] * self.N)  # Previous strain field (for semi-discretization)
        mcE_pprev = np.array([mcE_star] * self.N)   # Previous strain field (for semi-discretization)
        mcV_prev = np.zeros((self.N,6))  # Previous strain field (for semi-discretization)
        mcV_pprev = np.zeros((self.N,6))  # Previous strain field (for semi-discretization)

        # g_SE_list = []  # List to store configurations at each time step
        # mcE_list = []  # List to store strain fields at each time step
        # Lambda_list = []  # List to store strain fields at each time step
        # Lambda0_list = []
        # mcV_list = []  # List to store velocity fields at each time step
        g_SE_list = np.zeros((self.N_steps,self.N,4,4))  # List to store configurations at each time step
        mcE_list = np.zeros((self.N_steps,self.N,6,))  # List to store strain fields at each time step
        Lambda_list = np.zeros((self.N_steps,self.N,6,))  # List to store strain fields at each time step
        Lambda0_list = np.zeros((self.N_steps,6,))
        mcV_list = np.zeros((self.N_steps,self.N,6,))  # List to store velocity fields at each time step
        # initial guess for stess field 
        mcE0=np.zeros((6,))
        # parameters
        mbM0, mbK0, mbD0 = self.soft_robot_parameters()
        
        tendon_forces = np.dot(self.tendon_tensions, self.tendon_dirs) 

        Lambda0_initial_guess = np.zeros((6,))

        # Integrate the rod over time
        for i in range(self.N_steps):
            # solve the BVP
            result, solution = self.solve_bvp_shooting_method(
            g_SE0, mcV0, mcE_prev, mcE_pprev, mcV_prev, mcV_pprev,
            s_points, target_Lambda_tip, tendon_forces,Lambda0_initial_guess=Lambda0_initial_guess,
            method='least_squares', verbose=True)

            g_SE3 = solution['g_SE']
            mcE = solution['mcE']
            mcV = solution['mcV']
            Lambda = solution['Lambda']
            optimal_Lambda0 = result.x
            # solution['Lambda_prime']
            # solution['mcV_prime']
            # reset the target boundary conditions
            target_Lambda_tip = np.zeros((6,))
            # Store the results
            g_SE_list[i,:,:,:] = g_SE3
            mcE_list[i,:,:] = mcE
            mcV_list[i,:,:] = mcV
            Lambda_list[i,:,:] = Lambda
            Lambda0_list[i,:] = optimal_Lambda0
            
            # Update initial conditions for the next time step (s = 0)
            mcE0 = mcE[0,:]
            mcV0 = mcV[0,:]
            g_SE0 = g_SE3[0,:,:]
            Lambda0_initial_guess = optimal_Lambda0
            
            # Store previous states for the next iteration
            mcE_pprev = mcE_prev.copy()
            mcE_prev = mcE.copy()
            mcV_pprev = mcV_prev.copy()
            mcV_prev = mcV.copy()
            
            t0 += self.dt
            print('t: ',t0)
            tspan.append(t0)
            # Visualize the configuration
            # visualize_rod(g_SE3, mcV=mcV,mcE=mcE)
            # self.plot_shooting_results(solution, target_Lambda_tip)

            # check if the rod has moved 
            # print('g_SE@ g_SE_inv:', g_SE[-1,:,:]@SE3_inv(g_SE_list[0][-1,:,:]))
    #         plot_shooting_results(solution, target_Lambda_tip)
        
        return g_SE_list, mcE_list, mcV_list, Lambda_list, Lambda0_list
    


    def time_integrate_cosserat_rod_over_controls(self, g_SE0, mcV0, target_Lambda_tip, tendon_forces=None, verbose=False):
        # integrate the rod over time 
        tspan = [self.t0]
        t0 = self.t0
        # interval for rod 
        s_points = np.linspace(0, self.L, self.N)
        # Initial conditions
        mcE_star = self.mcE_rest(0.0)  # Rest strain field
        # Initial conditions 
        # mcE0 = np.linalg.inv(mbK) @ Lambda0 + mcE_star  # Initial strain field
        # mcE_star = mcE0
        mcE_prev = np.array([mcE_star] * self.N)  # Previous strain field (for semi-discretization)
        mcE_pprev = np.array([mcE_star] * self.N)   # Previous strain field (for semi-discretization)
        mcV_prev = np.zeros((self.N,6))  # Previous strain field (for semi-discretization)
        mcV_pprev = np.zeros((self.N,6))  # Previous strain field (for semi-discretization)

        # g_SE_list = []  # List to store configurations at each time step
        # mcE_list = []  # List to store strain fields at each time step
        # Lambda_list = []  # List to store strain fields at each time step
        # Lambda0_list = []
        # mcV_list = []  # List to store velocity fields at each time step
        g_SE_list = np.zeros((self.N_steps,self.N,4,4))  # List to store configurations at each time step
        mcE_list = np.zeros((self.N_steps,self.N,6,))  # List to store strain fields at each time step
        Lambda_list = np.zeros((self.N_steps,self.N,6,))  # List to store strain fields at each time step
        Lambda0_list = np.zeros((self.N_steps,6,))
        mcV_list = np.zeros((self.N_steps,self.N,6,))  # List to store velocity fields at each time step
        # initial guess for stess field 
        mcE0=np.zeros((6,))
        # parameters
        mbM0, mbK0, mbD0 = self.soft_robot_parameters()
         
        Lambda0_initial_guess = np.zeros((6,))

        # Integrate the rod over time
        for t in range(self.N_steps):
            # Get tendon forces for current time step
            if callable(tendon_forces):
                current_tendon_forces = tendon_forces(t * self.dt)
            elif tendon_forces is not None and tendon_forces.shape == (self.N_steps, 3):
                current_tendon_forces = tendon_forces[t]
            else:
                current_tendon_forces = tendon_forces if tendon_forces is not None else np.zeros((3,))

            # Integrate one time step
            result, solution = self.solve_bvp_shooting_method(
                g_SE0, mcV0, mcE_prev, mcE_pprev, mcV_prev, mcV_pprev,
                s_points, target_Lambda_tip, current_tendon_forces,Lambda0_initial_guess=Lambda0_initial_guess,
                method='least_squares', verbose=True)

            g_SE3 = solution['g_SE']
            mcE = solution['mcE']
            mcV = solution['mcV']
            Lambda = solution['Lambda']
            optimal_Lambda0 = result.x
            # solution['Lambda_prime']
            # solution['mcV_prime']
            # reset the target boundary conditions
            
            # Store the results
            g_SE_list[t,:,:,:] = g_SE3
            mcE_list[t,:,:] = mcE
            mcV_list[t,:,:] = mcV
            Lambda_list[t,:,:] = Lambda
            Lambda0_list[t,:] = optimal_Lambda0
            
            # Update initial conditions for the next time step (s = 0)
            mcE0 = mcE[0,:]
            mcV0 = mcV[0,:]
            g_SE0 = g_SE3[0,:,:]
            Lambda0_initial_guess = optimal_Lambda0
            
            # Store previous states for the next iteration
            mcE_pprev = mcE_prev.copy()
            mcE_prev = mcE.copy()
            mcV_pprev = mcV_prev.copy()
            mcV_prev = mcV.copy()
            
            t0 += self.dt
            print('t: ',t0)
            tspan.append(t0)
            # Visualize the configuration
            # visualize_rod(g_SE3, mcV=mcV,mcE=mcE)
            # self.plot_shooting_results(solution, target_Lambda_tip)

            # check if the rod has moved 
            # print('g_SE@ g_SE_inv:', g_SE[-1,:,:]@SE3_inv(g_SE_list[0][-1,:,:]))
    #         plot_shooting_results(solution, target_Lambda_tip)
        
        return g_SE_list, mcE_list, mcV_list, Lambda_list, Lambda0_list
    




    # ================================================================== Plotting Visualization
    def plot_shooting_results(self,solution, target_Lambda_tip=None):
        """
        Plot the results of the shooting method solution.
        
        Args:
            solution: Solution dictionary from solve_bvp_shooting_method
            target_Lambda_tip: Target tip stress for comparison
        """
        
        s_points = solution['s_points']
        Lambda = solution['Lambda']
        mcE = solution['mcE']
        mcV = solution['mcV']
        g_SE = solution['g_SE']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot stress components
        axes[0, 0].plot(s_points, Lambda[:, :3], label=['τ₁', 'τ₂', 'τ₃'])
        axes[0, 0].set_title('Moment Components')
        axes[0, 0].set_xlabel('Arc length s')
        axes[0, 0].set_ylabel('Moment')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(s_points, Lambda[:, 3:], label=['F₁', 'F₂', 'F₃'])
        axes[0, 1].set_title('Force Components')
        axes[0, 1].set_xlabel('Arc length s')
        axes[0, 1].set_ylabel('Force')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot strain components
        axes[0, 2].plot(s_points, mcE[:, :3], label=['κ₁', 'κ₂', 'κ₃'])
        axes[0, 2].set_title('Curvature Components')
        axes[0, 2].set_xlabel('Arc length s')
        axes[0, 2].set_ylabel('Curvature')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        axes[1, 0].plot(s_points, mcE[:, 3:], label=['γ₁', 'γ₂', 'γ₃'])
        axes[1, 0].set_title('Shear/Extension Components')
        axes[1, 0].set_xlabel('Arc length s')
        axes[1, 0].set_ylabel('Strain')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot rod configuration (centerline)
        positions = g_SE[:, :3, 3]
        axes[1, 1].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
        axes[1, 1].set_title('Rod Centerline (XY view)')
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Y')
        axes[1, 1].axis('equal')
        axes[1, 1].grid(True)
        
        # Plot 3D configuration
        ax_3d = fig.add_subplot(2, 3, 6, projection='3d')
        ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
        ax_3d.set_title('Rod Centerline (3D)')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        
        # Add target tip stress comparison if provided
        if target_Lambda_tip is not None:
            final_Lambda = Lambda[-1]
            print("\nBoundary condition comparison:")
            print(f"Target tip stress: {self.target_Lambda_tip}")
            print(f"Achieved tip stress: {final_Lambda}")
            print(f"Error: {final_Lambda - self.target_Lambda_tip}")
            print(f"Error norm: {np.linalg.norm(final_Lambda - self.target_Lambda_tip)}")
        
        plt.tight_layout()
        plt.show()

    
    def animate_cosserat_rod(self,g_SE_list, mcE_list, mcV_list, Lambda_list, 
                        rigid_motion_axis=None, set_axes_equal=None):
        """
        Animate the Cosserat rod simulation results
        
        Parameters:
        - g_SE_list: (N_steps, N, 4, 4) - SE(3) transformations over time
        - mcE_list: (N_steps, N, 6) - strain fields over time  
        - mcV_list: (N_steps, N, 6) - velocity fields over time
        - Lambda_list: (N_steps, N, 6) - force/moment fields over time
        - dt: time step
        - rigid_motion_axis: function to extract axes from SE(3) matrix
        - set_axes_equal: function to set equal axis scaling
        """
        
        N_steps, N, _, _ = g_SE_list.shape
        
        # Create figure and axis
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Pre-compute rod backbone positions for all time steps
        all_positions = np.zeros((N_steps, N, 3))
        
        for t in range(N_steps):
            for i in range(N):
                # Extract position from SE(3) matrix
                all_positions[t, i, :] = g_SE_list[t, i, :3, 3]
        
        # Set up plot limits based on all data
        all_pos_flat = all_positions.reshape(-1, 3)
        margin = 0.2
        x_range = [all_pos_flat[:, 0].min() - margin, all_pos_flat[:, 0].max() + margin]
        y_range = [all_pos_flat[:, 1].min() - margin, all_pos_flat[:, 1].max() + margin]
        z_range = [all_pos_flat[:, 2].min() - margin, all_pos_flat[:, 2].max() + margin]
        
        def animate(frame):
            ax.clear()
            
            # Current time step
            t = frame
            current_time = t * self.dt
            
            # Extract current rod configuration
            positions = all_positions[t]
            mcE = mcE_list[t]
            mcV = mcV_list[t]
            Lambda = Lambda_list[t]
            
            # Plot rod backbone
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'ko-', linewidth=3, markersize=6, label='Rod backbone')
            
            # Plot coordinate frames along the rod (every few points to avoid clutter)
            step = max(1, N // 8)  # Show ~8 coordinate frames
            for i in range(0, N, step):
                pos = positions[i]
                
                if rigid_motion_axis is not None:
                    try:
                        x_axis, y_axis, z_axis, _ = rigid_motion_axis(g_SE_list[t, i])
                        scale = 0.1
                        ax.quiver(*pos, *(x_axis * scale), color='red', alpha=0.7, arrow_length_ratio=0.15)
                        ax.quiver(*pos, *(y_axis * scale), color='green', alpha=0.7, arrow_length_ratio=0.15)
                        ax.quiver(*pos, *(z_axis * scale), color='blue', alpha=0.7, arrow_length_ratio=0.15)
                    except:
                        # Fallback: extract rotation matrix directly
                        R = g_SE_list[t, i, :3, :3]
                        scale = 0.1
                        ax.quiver(*pos, *(R[:, 0] * scale), color='red', alpha=0.7, arrow_length_ratio=0.15)
                        ax.quiver(*pos, *(R[:, 1] * scale), color='green', alpha=0.7, arrow_length_ratio=0.15)
                        ax.quiver(*pos, *(R[:, 2] * scale), color='blue', alpha=0.7, arrow_length_ratio=0.15)
            
            # Plot velocity field (every few points)
            for i in range(0, N, step):
                pos = positions[i]
                velocity = mcV[i, 3:6]  # Linear velocity
                angular_vel = mcV[i, :3]  # Angular velocity
                
                if np.linalg.norm(velocity) > 1e-6:
                    ax.quiver(*pos, *velocity, color='purple', alpha=0.8, 
                            arrow_length_ratio=0.1, linewidth=2)
                if np.linalg.norm(angular_vel) > 1e-6:
                    ax.quiver(*pos, *angular_vel, color='orange', alpha=0.8, 
                            arrow_length_ratio=0.1, linewidth=2)
            
            # Plot strain field (every few points)
            for i in range(0, N, step):
                pos = positions[i]
                strain_linear = mcE[i, 3:6]  # Linear strain
                strain_angular = mcE[i, :3]  # Angular strain
                
                if np.linalg.norm(strain_linear) > 1e-6:
                    ax.quiver(*pos, *strain_linear, color='cyan', alpha=0.8, 
                            arrow_length_ratio=0.15, linewidth=2)
                if np.linalg.norm(strain_angular) > 1e-6:
                    ax.quiver(*pos, *strain_angular, color='pink', alpha=0.8, 
                            arrow_length_ratio=0.15, linewidth=2)
            
            # Highlight tip of the rod
            tip_pos = positions[-1]
            ax.scatter(*tip_pos, color='red', s=100, alpha=0.9, label='Rod tip')
            
            # Show trajectory trail of the tip
            if t > 0:
                tip_trail = all_positions[:t+1, -1, :]
                ax.plot(tip_trail[:, 0], tip_trail[:, 1], tip_trail[:, 2], 
                    'r--', alpha=0.5, linewidth=2, label='Tip trajectory')
            
            # Set plot properties
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_zlim(z_range)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            if set_axes_equal is not None:
                set_axes_equal(ax)
            
            ax.set_title(f'Cosserat Rod Dynamics\nTime: {current_time:.3f}s (Step {t+1}/{N_steps})')
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            
            return []
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=N_steps, 
                                    interval=max(50, int(self.dt*1000)), blit=False, repeat=True)
        
        return fig, anim



    def analyze_rod_dynamics(self,g_SE_list, mcE_list, mcV_list):
        """
        Create analysis plots of the rod dynamics
        """
        N_steps, N = g_SE_list.shape[:2]
        time_array = np.arange(N_steps) * self.dt
        
        # Extract key metrics
        tip_positions = np.zeros((N_steps, 3))
        tip_velocities = np.zeros((N_steps, 3))
        rod_length = np.zeros(N_steps)
        
        for t in range(N_steps):
            tip_positions[t] = g_SE_list[t, -1, :3, 3]
            tip_velocities[t] = mcV_list[t, -1, 3:6]
            
            # Calculate rod length (arc length)
            positions = g_SE_list[t, :, :3, 3]
            differences = np.diff(positions, axis=0)
            segment_lengths = np.linalg.norm(differences, axis=1)
            rod_length[t] = np.sum(segment_lengths)
        
        # Create analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Tip position
        axes[0,0].plot(time_array, tip_positions[:, 0], 'r-', label='X')
        axes[0,0].plot(time_array, tip_positions[:, 1], 'g-', label='Y')  
        axes[0,0].plot(time_array, tip_positions[:, 2], 'b-', label='Z')
        axes[0,0].set_title('Tip Position vs Time')
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Position')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Tip velocity
        axes[0,1].plot(time_array, tip_velocities[:, 0], 'r-', label='Vx')
        axes[0,1].plot(time_array, tip_velocities[:, 1], 'g-', label='Vy')
        axes[0,1].plot(time_array, tip_velocities[:, 2], 'b-', label='Vz')
        axes[0,1].set_title('Tip Velocity vs Time')
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Velocity')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Rod length
        axes[1,0].plot(time_array, rod_length, 'k-', linewidth=2)
        axes[1,0].set_title('Rod Length vs Time')
        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('Length')
        axes[1,0].grid(True, alpha=0.3)
        
        # Tip trajectory in 3D
        axes[1,1].remove()
        ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
        ax_3d.plot(tip_positions[:, 0], tip_positions[:, 1], tip_positions[:, 2], 
                'b-', linewidth=2, alpha=0.7)
        ax_3d.scatter(tip_positions[0, 0], tip_positions[0, 1], tip_positions[0, 2], 
                    color='green', s=100, label='Start')
        ax_3d.scatter(tip_positions[-1, 0], tip_positions[-1, 1], tip_positions[-1, 2], 
                    color='red', s=100, label='End')
        ax_3d.set_title('3D Tip Trajectory')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.legend()
        
        plt.tight_layout()
        return fig

    def plot_shooting_results(self,solution, target_Lambda_tip=None):
        """
        Plot the results of the shooting method solution.
        
        Args:
            solution: Solution dictionary from solve_bvp_shooting_method
            target_Lambda_tip: Target tip stress for comparison
        """
        
        s_points = solution['s_points']
        Lambda = solution['Lambda']
        mcE = solution['mcE']
        mcV = solution['mcV']
        g_SE = solution['g_SE']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot stress components
        axes[0, 0].plot(s_points, Lambda[:, :3], label=['τ₁', 'τ₂', 'τ₃'])
        axes[0, 0].set_title('Moment Components')
        axes[0, 0].set_xlabel('Arc length s')
        axes[0, 0].set_ylabel('Moment')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(s_points, Lambda[:, 3:], label=['F₁', 'F₂', 'F₃'])
        axes[0, 1].set_title('Force Components')
        axes[0, 1].set_xlabel('Arc length s')
        axes[0, 1].set_ylabel('Force')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot strain components
        axes[0, 2].plot(s_points, mcE[:, :3], label=['κ₁', 'κ₂', 'κ₃'])
        axes[0, 2].set_title('Curvature Components')
        axes[0, 2].set_xlabel('Arc length s')
        axes[0, 2].set_ylabel('Curvature')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        axes[1, 0].plot(s_points, mcE[:, 3:], label=['γ₁', 'γ₂', 'γ₃'])
        axes[1, 0].set_title('Shear/Extension Components')
        axes[1, 0].set_xlabel('Arc length s')
        axes[1, 0].set_ylabel('Strain')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot rod configuration (centerline)
        positions = g_SE[:, :3, 3]
        axes[1, 1].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
        axes[1, 1].set_title('Rod Centerline (XY view)')
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Y')
        axes[1, 1].axis('equal')
        axes[1, 1].grid(True)
        
        # Plot 3D configuration
        ax_3d = fig.add_subplot(2, 3, 6, projection='3d')
        ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
        ax_3d.set_title('Rod Centerline (3D)')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        
        # Add target tip stress comparison if provided
        if target_Lambda_tip is not None:
            final_Lambda = Lambda[-1]
            print("\nBoundary condition comparison:")
            print(f"Target tip stress: {target_Lambda_tip}")
            print(f"Achieved tip stress: {final_Lambda}")
            print(f"Error: {final_Lambda - target_Lambda_tip}")
            print(f"Error norm: {np.linalg.norm(final_Lambda - target_Lambda_tip)}")
        
        plt.tight_layout()
        plt.show()


