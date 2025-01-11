import sys
import os
import string
import numpy as np
import matplotlib.pyplot as plt

# We had a problem with the path of the other module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#############
# Constants #
#############

nb_points = 100

R = 8.314 # Ideal gas constant in J/(mol* K)

M_N2=0.02802 #kg/mol

M_air=0.02895 #kg/mol

M_CH4=0.016042 #kg/mol

dt = 1e-10

Lx = 2e-3

Ly = 2e-3

L_slot = 0.5e-3

L_coflow = 0.5e-3

U_slot = 1.0

T_slot = 300

U_coflow = 0.2

T_coflow = 300

# Fluid density
rho = 1.1614

# Kinematic viscosity
eta = 15e-6

dx = Lx / (nb_points - 1)

dy = Ly / (nb_points - 1)

nu = 15e-6

tolerance = 1e-6

max_iter = 1000 # Maximum number of iterations

omega = 1.5 # Parameter for the Successive Overrelaxation Method (SOR), it has to be between 1 and 2


##################
# Initial fields #
##################

# Velocity fields
v = np.zeros((nb_points, nb_points))
u = np.zeros((nb_points, nb_points))

# Pressure field
P = np.zeros((nb_points, nb_points)) 
    
P[0, 0:int(L_slot/Lx * nb_points)] = rho * R * T_slot/M_CH4
P[-1, 0:int(L_slot/Lx * nb_points)] = rho * R * T_slot/M_air
P[0, int(L_slot/Lx * nb_points):int((L_slot + L_coflow)/Lx * nb_points)] = rho * R * T_coflow/M_N2
P[-1, int(L_slot/Lx * nb_points):int((L_slot + L_coflow)/Lx * nb_points)] = rho * R * T_coflow/M_N2
#################################################
# Derivatives and second derivatives definition #
#################################################
def derivative_x(u: np.array, var_name : string, dx=dx) -> np.array:
    """
    Vectorized computation of the derivative along x-direction
    with upwinding based on the sign of u.
    
    Parameters:
        u (np.array): 2D array of values.
        dx (float): Grid spacing in the x-direction.
    
    Returns:
        np.array: Array of derivatives.
    """
    # Create shifted versions of the array for upwind calculation
    u_left = np.roll(u, shift=1, axis=0)   # u[i-1, j]
    u_right = np.roll(u, shift=-1, axis=0) # u[i+1, j]
    
    # Apply upwind scheme based on the sign of u

    derivative = np.where(u > 0, (u - u_left) / dx, (u_right - u) / dx)

    if var_name == "u":
        derivative[:,-1] = 0 # Right free limit condition
 
    if var_name == "v": 
        derivative[:,0] = 0 # Left slippery wall
        derivative[:,-1] = 0 # Right free limit condition
    
    if var_name == "P":
        derivative[:,0] = 0 # Left slippery wall

    return derivative

def derivative_y(u: np.array, var_name : string, dy=dy) -> np.array:
    """
    Vectorized computation of the derivative along y-direction
    with upwinding based on the sign of u.
    
    Parameters:
        u (np.array): 2D array of values.
        dy (float): Grid spacing in the y-direction.
    
    Returns:
        np.array: Array of derivatives.
    """
    # Create shifted versions of the array for upwind calculation
    u_left = np.roll(u, shift=1, axis=1)   # u[i, j - 1]
    u_right = np.roll(u, shift=-1, axis=1) # u[i, j + 1]
    
    # Apply upwind scheme based on the sign of u
    derivative = np.where(u > 0, (u - u_left) / dy, (u_right - u) / dy)

    if var_name == "P":
        derivative[0,:] = 0
        derivative[-1,:] = 0

    return derivative

def second_centered_x(u:np.array, dx = dx) -> np.array: 

    u_left = np.roll(u, shift=1, axis=0)   # u[i-1, j]
    u_right = np.roll(u, shift=-1, axis=0) # u[i+1, j]
    
    return (u_right - 2 * u + u_left) / dx**2

def second_centered_y(u:np.array, dy = dy) -> np.array: 

    u_left = np.roll(u, shift=1, axis=1)   # u[i, j - 1]
    u_right = np.roll(u, shift=-1, axis=1) # u[i, j + 1]

    return (u_right - 2 * u + u_left) / dy**2

import numpy as np

# def sor(P: np.array, f: np.array, dx=dx, dy=dy, omega=omega, tol=1e-2, max_iter=1000) -> np.array:
#     """
#     Solve the Poisson equation for pressure correction using Successive Over-Relaxation (SOR).

#     Parameters:
#         P (np.array): Initial pressure field (2D array).
#         f (np.array): Source term (right-hand side of the Poisson equation).
#         dx (float): Grid spacing in the x-direction.
#         dy (float): Grid spacing in the y-direction.
#         omega (float): Relaxation factor for SOR (1 < omega < 2).
#         tol (float): Convergence tolerance for the residual norm.
#         max_iter (int): Maximum number of iterations.

#     Returns:
#         np.array: Updated pressure field.
#     """
#     coef = 1 / (2 / dx**2 + 2 / dy**2)  # Coefficient for the discretized Poisson equation
#     Nx, Ny = P.shape  # Grid size

#     for it in range(max_iter):
#         P_old = P.copy()  # Save the old pressure field
#         P_new = np.zeros_like(P)
#         # Update pressure using SOR
#         for i in range(1, Nx - 1):  # Exclude boundary points
#             for j in range(1, Ny - 1):
#                 # Compute the new pressure using the SOR formula                
#                 P_new = coef * (
#                     (P[i+1, j] + P[i-1, j]) / dx**2 +
#                     (P[i, j+1] + P[i, j-1]) / dy**2 -
#                     f[i, j]
#                 )
#                 P[i, j] = (1 - omega) * P[i, j] + omega * P_new

#         # Compute the residual norm
#         residual = np.zeros_like(P)
#         for i in range(1, Nx - 1):
#             for j in range(1, Ny - 1):
#                 residual[i, j] = f[i, j] - (
#                     (P[i+1, j] - 2 * P[i, j] + P[i-1, j]) / dx**2 +
#                     (P[i, j+1] - 2 * P[i, j] + P[i, j-1]) / dy**2
#                 )
#         residual_norm = np.linalg.norm(residual, ord=np.inf)

#         # Check for convergence
#         if residual_norm < tol:
#             print(f"SOR converged after {it + 1} iterations with residual norm {residual_norm:.2e}.")
#             return P

#     print("SOR did not converge within the maximum number of iterations.")
#     return P


def sor(P: np.array, f: np.array, dx=dx, dy=dy, omega=omega, tol=10e-2, max_iter=max_iter) -> np.array:
    """
    Solve the Poisson equation for pressure correction using Successive Over-Relaxation (SOR).

    Parameters:
        P (np.array): Initial pressure field (2D array).
        f (np.array): Source term (right-hand side of the Poisson equation).
        omega (float): Relaxation factor for SOR (1 < omega < 2).
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.

    Returns:
        np.array: Updated pressure field.
    """
   
    coef = 1 / (2/(dx**2) + 2/(dy**2))  # Coefficient for the discretized Poisson equation

    for it in range(max_iter):
        P_old = P.copy()  # Save the old pressure field for convergence check

        # Update pressure using SOR
        P_new = np.zeros_like(P)

        for i in range(1, nb_points - 1):  # Exclude boundary points
            for j in range(1, nb_points - 1):
                # Compute the new pressure using the SOR formula                
                P_new[i,j] = coef * ((P_old[i+1,j] + P_new[i-1,j])/dx**2 + (P_old[i,j+1] + P_new[i,j-1])/dy**2 - f[i,j])

                P[i, j] = (1 - omega) * P[i, j] + omega * P_new[i,j]

        # Check for convergence using the infinity norm of the difference
        if np.linalg.norm(P - P_old, ord=np.inf) < tol: # Takes the maximum difference amongst all the points
            print(f"SOR converged after {it+1} iterations.")
            return P

    print("SOR did not converge within the maximum number of iterations.")
    return P


u[:,0] = 0
u[0,:] = 0
u[-1,:] =0
# Boundary conditions for u, v and their first derivatives (the latter is included inside the derivative functions)
v[0, 0:int(L_slot/Lx * nb_points)] = -U_slot # Speed at the top of the "flow" region (Methane)
v[-1, 0:int(L_slot/Lx * nb_points)] = U_slot # Speed at the bottom of the "flow" region (Air)
v[0, int(L_slot/Lx * nb_points):int((L_slot + L_coflow)/Lx * nb_points)] = - U_coflow # Speed at the top of the "coflow" region (Nitrogen)
v[-1, int(L_slot/Lx * nb_points):int((L_slot + L_coflow)/Lx * nb_points)] = U_coflow
##########################
# Fractional-step method #
##########################

for it in range(max_iter):

      # Speed at the top of the "coflow" region (Nitrogen)

    # Step 1
    u_star = u - dt * (u * derivative_x(u, "u") + v * derivative_y(u, "u"))
    v_star = v - dt * (u * derivative_x(v, "v") + v * derivative_y(v, "v"))

    u_star[:,0] = 0
    u_star[0,:] = 0
    u_star[-1,:] =0
    # Boundary conditions for u_star and v_star
    v_star[0, 0:int(L_slot/Lx * nb_points)] = -U_slot # Speed at the top of the "flow" region (Methane)
    v_star[-1, 0:int(L_slot/Lx * nb_points)] = U_slot # Speed at the bottom of the "flow" region (Air)
    v_star[0, int(L_slot/Lx * nb_points):int((L_slot + L_coflow)/Lx * nb_points)] = - U_coflow # Speed at the top of the "coflow" region (Nitrogen)
    v_star[-1, int(L_slot/Lx * nb_points):int((L_slot + L_coflow)/Lx * nb_points)] = U_coflow  # Speed at the top of the "coflow" region (Nitrogen)

    # Step 2 
    u_double_star = u_star + dt * (nu * (second_centered_x(u_star) + second_centered_y(u_star)))
    v_double_star = v_star + dt * (nu * (second_centered_x(v_star) +second_centered_y(v_star)))

    u_double_star[:,0] = 0
    u_double_star[0,:] = 0
    u_double_star[-1,:] =0
    # Boundary conditions for P, u_double_star, v_double_star, d(u_double_star)/dx and d(v_double_star)/dy (the latter two are directly included in the derivative functions)
    v_double_star[0, 0:int(L_slot/Lx * nb_points)] = -U_slot # Speed at the top of the "flow" region (Methane)
    v_double_star[-1, 0:int(L_slot/Lx * nb_points)] = U_slot # Speed at the bottom of the "flow" region (Air)
    v_double_star[0, int(L_slot/Lx * nb_points):int((L_slot + L_coflow)/Lx * nb_points)] = - U_coflow # Speed at the top of the "coflow" region (Nitrogen)
    v_double_star[-1, int(L_slot/Lx * nb_points):int((L_slot + L_coflow)/Lx * nb_points)] = U_coflow  # Speed at the top of the "coflow" region (Nitrogen)
    
    # Step 3
    P = sor(P, f = rho/dt * (derivative_x(u_double_star, "u") + derivative_y(v_double_star, "v")))

    # Boundary conditions for P (and for dP/dx and dP/dy, that are already inside the derivative functions)
    P[0, 0:int(L_slot/Lx * nb_points)] = rho * R * T_slot/M_CH4
    P[-1, 0:int(L_slot/Lx * nb_points)] = rho * R * T_slot/M_air
    P[0, int(L_slot/Lx * nb_points):int((L_slot + L_coflow)/Lx * nb_points)] = rho * R * T_coflow/M_N2
    P[-1, int(L_slot/Lx * nb_points):int((L_slot + L_coflow)/Lx * nb_points)] = rho * R * T_coflow/M_N2

    # Step 4
    u_new = u_double_star - dt/rho * derivative_x(P, "P")
    v_new = v_double_star - dt/rho * derivative_y(P, "P")

    u_new[:,0] = 0
    u_new[0,:] = 0
    u_new[-1,:] =0
    # Boundary conditions for P, u_new, v_new, d(u_new)/dx and d(v_new)/dy (the latter two are directly included in the derivative functions)
    v_new[0, 0:int(L_slot/Lx * nb_points)] = -U_slot # Speed at the top of the "flow" region (Methane)
    v_new[-1, 0:int(L_slot/Lx * nb_points)] = U_slot # Speed at the bottom of the "flow" region (Air)
    v_new[0, int(L_slot/Lx * nb_points):int((L_slot + L_coflow)/Lx * nb_points)] = - U_coflow # Speed at the top of the "coflow" region (Nitrogen)
    v_new[-1, int(L_slot/Lx * nb_points):int((L_slot + L_coflow)/Lx * nb_points)] = U_coflow

    # Updating of the new fields
    u = np.copy(u_new)
    v = np.copy(v_new)


# Once the u field is correctly advected, we calculate "a"
strain_rate = np.abs(-derivative_x(u, "u")) # Since dv/dy = -du/dx

max_strain_rate = np.max(strain_rate)

print(f"Maximum strain rate (|∂v/∂y|) on left wall: {max_strain_rate:.6f} s^-1")