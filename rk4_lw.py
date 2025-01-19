import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
import derivatives_vectorized as der
import plotting as plots
from constants import (
    nb_points, dt, nb_timesteps, Lx, Ly, L_slot, L_coflow, U_slot, U_coflow, 
    T_slot, T_coflow, dx, dy, max_iter_sor, omega, tolerance_sor, tolerance_sys, 
    rho, nu, A, T_a, c_p, W_N2, W_O2, W_CH4, W_H2O, W_CO2, 
    nu_ch4, nu_o2, nu_n2, nu_h2o, nu_co2, h_n2, h_o2, h_ch4, h_h2o, h_co2
)

from pathlib import Path
#################################################
# Input on image storage path and figure saving #
#################################################

# Path where the images will be stored (the name of the folder is specified at the end of the string)
output_folder  = str(Path(__file__).parent/ Path("RESULTS")/Path("result_1")) #'C:\\Users\\danie\\Desktop\\Code results\\run_12' 
dpi = 100 # For storing the images with high quality
show_figures = False # If this variable is set to false, all the images are stored in the selected path and are not shown here
compute_animations = True

# If this variable is set to true, the calculation will stop to evolve when the tolerance for convergence is reached.
# If it is set to False, the calculation will stop at the given final_time
convergence_by_tolerance = False 

# To make sure that the folder exists
os.makedirs(output_folder, exist_ok=True) 


##################
# Initial fields #
##################

# Velocity fields
v = np.zeros((nb_points, nb_points))
u = np.zeros((nb_points, nb_points))

# Pressure field
P = np.zeros((nb_points, nb_points))
P[:, :] = 101325

# Temperature field
T = np.zeros((nb_points, nb_points))
T[:, :] = 300 # We apply a constant field everywhere before applying the boundary conditions to avoid division by 0 in the exponential
T[int(3 * nb_points / 8) : int(5 * nb_points / 8), 1 :] = 1000

# Species fields
Y_n2 = np.zeros((nb_points, nb_points))
Y_o2 = np.zeros((nb_points, nb_points)) 
Y_ch4 = np.zeros((nb_points, nb_points)) 
Y_h2o = np.zeros((nb_points, nb_points))
Y_co2 = np.zeros((nb_points, nb_points)) 


##########################################
# Successive Overrelaxation (SOR) method #
##########################################

@jit(fastmath=True, nopython=True, cache=False)
def sor(P:np.array, f:np.array, tolerance_sor:float=tolerance_sor, max_iter_sor:float=max_iter_sor, omega:float=omega):
    """
    Successive Overrelaxation (SOR) method for solving the pressure Poisson equation.
    Optimized using Numba

    Parameters:
        P (np.array): Initial guess for the pressure field.
        f (np.array): Right-hand side of the Poisson equation.
        tolerance (float): Convergence tolerance for the iterative method.
        max_iter_sor (int): Maximum number of iterations.
        omega (float): Relaxation factor, between 1 and 2.

    Returns:
        np.array: Updated pressure field.
    """
    dx2=dx**2
    dy2=dy**2
    coef = 2 * (1 / dx2 + 1 / dy2)

    for _ in range(max_iter_sor):
        P_old = P.copy()
        laplacian_P = np.zeros_like(P)
        
        for i in range(1, nb_points - 1):
            for j in range(1, nb_points - 1):
                laplacian_P[i, j] = (P_old[i+1, j] + P[i-1, j]) / dy2 + (P_old[i, j+1] + P[i, j-1]) / dx2
                
                # Update P using the SOR formula
                P[i, j] = (1 - omega) * P_old[i, j] + (omega / coef) * (laplacian_P[i, j] - f[i, j])

        P[:, 0] = 0
        P[:, 1] = P[:, 2] # dP/dx = 0 at the left wall
        P[0, 1:] = P[1, 1:] # dP/dy = 0 at the top wall
        P[-1, 1:] = P[-2, 1:] # dP/dy = 0 at the bottom wall
        P[:, -1] = 0 # P = 0 at the right free limit
        
        # Compute the residual to check for convergence
        residual = np.linalg.norm(P - P_old, ord=2)
        if np.linalg.norm(P_old, ord=2) > 10e-10:  # Avoid divide-by-zero by checking the norm
            residual /= np.linalg.norm(P_old, ord=2)

        if residual < tolerance_sor:
            break

    return P



# Vectorized SOR function
# @jit(fastmath=True, nopython=True, parallel=True)
# def sor(P, f, tolerance_sor=tolerance_sor, max_iter_sor=max_iter_sor, omega=omega):
#     """
#     Red-Black Successive Over-Relaxation (SOR) method for solving the pressure Poisson equation.
#     Optimized using Numba for parallel execution.

#     Parameters:
#         P (np.array): Initial guess for the pressure field.
#         f (np.array): Right-hand side of the Poisson equation.
#         tolerance_sor (float): Convergence tolerance for the iterative method.
#         max_iter_sor (int): Maximum number of iterations.
#         omega (float): Relaxation factor, between 1 and 2.

#     Returns:
#         np.array: Updated pressure field.
#     """
#     coef = 2 * (1 / dx**2 + 1 / dy**2)
#     nb_points = P.shape[0]

#     for _ in range(max_iter_sor):
#         P_old = P.copy()
#         max_diff = 0.0
#         # Red cells update
#         for j in prange(1, nb_points - 1):
#             for i in range(1 + (j % 2), nb_points - 1, 2):  # Alternate cells
#                 # Corrected laplacian: use both P_old and P
#                 laplacian_P = (P_old[i+1, j] + P[i-1, j]) / dy**2 + (P_old[i, j+1] + P[i, j-1]) / dx**2
#                 P[i, j] = (1 - omega) * P_old[i, j] + (omega / coef) * (laplacian_P - f[i, j])

#         # Black cells update
#         for j in prange(1, nb_points - 1):
#             for i in range(1 + ((j + 1) % 2), nb_points - 1, 2):  # Alternate cells
#                 # Corrected laplacian: use both P_old and P
#                 laplacian_P = (P_old[i+1, j] + P[i-1, j]) / dy**2 + (P_old[i, j+1] + P[i, j-1]) / dx**2
#                 P[i, j] = (1 - omega) * P_old[i, j] + (omega / coef) * (laplacian_P - f[i, j])

#         # Apply boundary conditions
#         P[:, 0] = 0
#         P[:, 1] = P[:, 2]  # dP/dx = 0 at the left wall
#         P[0, 1:] = P[1, 1:]  # dP/dy = 0 at the top wall
#         P[-1, 1:] = P[-2, 1:]  # dP/dy = 0 at the bottom wall
#         P[:, -1] = 0  # P = 0 at the right free limit
#         max_diff = np.max(np.abs(P - P_old))
#         if max_diff < tolerance_sor:
#             break
#         # Compute the residual for convergence
#         # residual = np.linalg.norm(P - P_old, ord=2)
#         # if np.linalg.norm(P_old, ord=2) > 1e-10:  # Avoid divide-by-zero
#         #     residual /= np.linalg.norm(P_old, ord=2)

#         # if residual < tolerance_sor:
#         #     break

#     return P



#######################
# Boundary conditions #
#######################

@jit(fastmath=True, nopython=True, cache=False)
def boundary_conditions(u:np.array, v:np.array, P:np.array, T:np.array, Y_n2:np.array, Y_o2:np.array, Y_ch4:np.array):
    # Boundary conditions for the velocity field
    u[:, 0] = 0
    u[:, 1] = 0 # Left slipping wall
    u[0, :] = 0 # Top non-slipping wall
    u[-1, :] = 0 # Bottom non-slipping wall
    u[:, -1] = u[:, -2] # du/dx = 0 at the right free limit

    v[:, 0] = 0 # Left wall
    v[0, 1 : int(L_slot / Lx * nb_points) + 1] = -U_slot # Speed at the top of the "flow" region (Methane)
    v[-1, 1 : int(L_slot / Lx * nb_points) + 1] = U_slot  # Speed at the bottom of the "flow" region (Air)
    v[0, int(L_slot / Lx * nb_points) + 1 : int((L_slot + L_coflow) / Lx * nb_points)] = -U_coflow  # Speed at the top of the "coflow" region (Nitrogen)
    v[-1, int(L_slot / Lx * nb_points) + 1 : int((L_slot + L_coflow) / Lx * nb_points)] = U_coflow # Speed at the bottom of the "coflow" region (Nitrogen)
    v[0, int((L_slot + L_coflow) / Lx * nb_points) : nb_points] = 0 # For the top non-slipping wall
    v[-1, int((L_slot + L_coflow) / Lx * nb_points) : nb_points] = 0  # For the bottom non-slipping wall 
    v[:, 1] = v[:, 2] # dv/dx = 0 at the left wall
    v[:, -1] = v[:, -2] # dv/dx = 0 at the right free limit

    T[0, 1 : int(L_slot / Lx * nb_points) + 1] = T_slot
    T[-1, 1 : int(L_slot / Lx * nb_points) + 1] = T_slot
    T[0, int(L_slot / Lx * nb_points) + 1 : int((L_slot + L_coflow) / Lx * nb_points)] = T_coflow
    T[-1, int(L_slot / Lx * nb_points) + 1 : int((L_slot + L_coflow) / Lx * nb_points)] = T_coflow

    P[:, 0] = 0
    P[:, 1] = P[:, 2] # dP/dx = 0 at the left wall
    P[0, 1:] = P[1, 1:] # dP/dy = 0 at the top wall
    P[-1, 1:] = P[-2, 1:] # dP/dy = 0 at the bottom wall
    P[:, -1] = 0 # P = 0 at the right free limit

    Y_n2[-1, 1 : int(L_slot / Lx * nb_points) + 1] = 0.767 # Initial condition for the nitrogen in air (bottom slot)
    Y_n2[0, int(L_slot / Lx * nb_points) : int((L_slot + L_coflow) / Lx * nb_points)] = 1
    Y_n2[-1, int(L_slot / Lx * nb_points) : int((L_slot + L_coflow) / Lx * nb_points)] = 1

    Y_o2[-1, 1 : int(L_slot / Lx * nb_points) + 1] = 0.233 # Initial condition for the oxigen in air (bottom slot)

    Y_ch4[0, 1 : int(L_slot / Lx * nb_points) + 1] = 1

    return u, v, P, T, Y_n2, Y_o2, Y_ch4


#########################
# First step of the RK4 #
#########################

@jit(fastmath=True, nopython=True, cache=False)
def compute_rhs_first_u(u : np.array, v : np.array):
    derivative_x = der.derivative_x_centered(u)
    derivative_y = der.derivative_y_centered(u)

    rhs = (
        u * derivative_x + v * derivative_y
    )
    
    return rhs


@jit(fastmath=True, nopython=True, cache=False)
def compute_rhs_first_v(u : np.array, v : np.array):
    derivative_x = der.derivative_x_centered(v)
    derivative_y = der.derivative_y_centered(v)

    rhs = (
        u * derivative_x + v * derivative_y
    )
    
    return rhs


@jit(fastmath=True, nopython=True, cache=False)
def rk4_first_step_frac_u(u : np.array, v : np.array, dt=dt):
    
    k1 = compute_rhs_first_u(u, v)
    k2 = compute_rhs_first_u(u + 0.5 * dt * k1, v)
    k3 = compute_rhs_first_u(u + 0.5 * dt * k2, v)
    k4 = compute_rhs_first_u(u + dt * k3, v)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


@jit(fastmath=True, nopython=True, cache=False)
def rk4_first_step_frac_v(u : np.array, v : np.array, dt:float=dt):
    
    k1 = compute_rhs_first_v(u, v)
    k2 = compute_rhs_first_v(u, v + 0.5 * dt * k1)
    k3 = compute_rhs_first_v(u, v + 0.5 * dt * k2)
    k4 = compute_rhs_first_v(u, v + dt * k3)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


##########################
# Second step of the RK4 #
##########################

@jit(fastmath=True, nopython=True, cache=False)
def compute_rhs_second(u : np.array):
    second_derivative_x = der.second_centered_x(u)
    second_derivative_y = der.second_centered_y(u)

    rhs = (
        nu * (second_derivative_x + second_derivative_y)
    )
    return rhs

@jit(fastmath=True, nopython=True, cache=False)
def rk4_second_step_frac(u : np.array, dt:float=dt):
    
    k1 = compute_rhs_second(u)
    k2 = compute_rhs_second(u + 0.5 * dt * k1)
    k3 = compute_rhs_second(u + 0.5 * dt * k2)
    k4 = compute_rhs_second(u + dt * k3)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


#################################################
# Fourth step of the RK4 (the third one is SOR) #
#################################################

@jit(fastmath=True, nopython=True, cache=False)
def compute_rhs_u(P : np.array):
    derivative_x = der.derivative_x_centered(P)

    rhs = (
        derivative_x
    )
    return rhs


@jit(fastmath=True, nopython=True, cache=False)
def compute_rhs_v(P:np.array):
    derivative_y = der.derivative_y_centered(P)

    rhs = (
        derivative_y
    )
    return rhs



@jit(fastmath=True, nopython=True, cache=False)
def rk4_fourth_step_frac_u(P : np.array, dt:float = dt):

    k1 = compute_rhs_u(P)
    k2 = compute_rhs_u(P + 0.5 * dt * k1)
    k3 = compute_rhs_u(P + 0.5 * dt * k2)
    k4 = compute_rhs_u(P + dt * k3)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


@jit(fastmath=True, nopython=True, cache=False)
def rk4_fourth_step_frac_v(P : np.array, dt:float = dt):
    
    k1 = compute_rhs_v(P)
    k2 = compute_rhs_v(P + 0.5 * dt * k1)
    k3 = compute_rhs_v(P + 0.5 * dt * k2)
    k4 = compute_rhs_v(P + dt * k3)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


#######################################
# Solving species transport using RK4 #
#######################################

@jit(fastmath=True, nopython=True, cache=False)
def compute_rhs_Y_k(Y_k:np.array, u : np.array, v : np.array, source_term: np.array):
    derivative_x = der.derivative_x_centered(Y_k)
    derivative_y = der.derivative_y_centered(Y_k)
    second_derivative_x = der.second_centered_x(Y_k)
    second_derivative_y = der.second_centered_y(Y_k)

    rhs = (
        -u * derivative_x
        - v * derivative_y
        + nu * (second_derivative_x + second_derivative_y)
        + source_term
    )
    return rhs


@jit(fastmath=True, nopython=True, cache=False)
def rk4_step_Y_k(Y_k:np.array, u : np. array, source_term: np.array, dt:float=dt):
    """
    Perform a single Runge-Kutta 4th order step for species advection-diffusion.

    Parameters:
        Y_k (np.array): Scalar field (e.g., species concentration).
        u, v (np.array): Velocity components.
        source_term (np.array): Source term for species.
        dx, dy, dt (float): Spatial and temporal resolutions.

    Returns:
        np.array: Updated scalar field after one RK4 step.
    """

    k1 = compute_rhs_Y_k(Y_k, u, v, source_term)
    k2 = compute_rhs_Y_k(Y_k + 0.5 * dt * k1, u, v, source_term)
    k3 = compute_rhs_Y_k(Y_k + 0.5 * dt * k2, u, v, source_term)
    k4 = compute_rhs_Y_k(Y_k + dt * k3, u, v, source_term)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


##########################
# Fractional-step method #
##########################
@jit(fastmath=True, nopython=True)
def system_evolution_kernel_lw(u:np.array, v:np.array, P:np.array, T:np.array, Y_n2:np.array, Y_o2:np.array, Y_ch4:np.array, Y_h2o:np.array, Y_co2:np.array):
    """
    Evolves the system using the Lax-Wendroff scheme instead of RK4.

    Parameters:
        u, v: Velocity fields (2D arrays).
        P: Pressure field (2D array).
        T: Temperature field (2D array).
        Y_n2, Y_o2, Y_ch4, Y_h2o, Y_co2: Species concentration fields (2D arrays).

    Returns:
        Updated fields after one timestep using Lax-Wendroff.
    """
    # Step 1: Apply boundary conditions
    u, v, P, T, Y_n2, Y_o2, Y_ch4 = boundary_conditions(u, v, P, T, Y_n2, Y_o2, Y_ch4)

    # Step 2: Compute half-step values for advection
    u_half = u - 0.5 * dt * (u * der.derivative_x_centered(u) + v * der.derivative_y_centered(u))
    v_half = v - 0.5 * dt * (u * der.derivative_x_centered(v) + v * der.derivative_y_centered(v))
    Y_n2_half = Y_n2 - 0.5 * dt * (u * der.derivative_x_centered(Y_n2) + v * der.derivative_y_centered(Y_n2))
    Y_o2_half = Y_o2 - 0.5 * dt * (u * der.derivative_x_centered(Y_o2) + v * der.derivative_y_centered(Y_o2))
    Y_ch4_half = Y_ch4 - 0.5 * dt * (u * der.derivative_x_centered(Y_ch4) + v * der.derivative_y_centered(Y_ch4))
    Y_h2o_half = Y_h2o - 0.5 * dt * (u * der.derivative_x_centered(Y_h2o) + v * der.derivative_y_centered(Y_h2o))
    Y_co2_half = Y_co2 - 0.5 * dt * (u * der.derivative_x_centered(Y_co2) + v * der.derivative_y_centered(Y_co2))

    # Step 3: Add diffusion terms to half-step values
    u_half += 0.5 * dt * nu * (der.second_centered_x(u) + der.second_centered_y(u))
    v_half += 0.5 * dt * nu * (der.second_centered_x(v) + der.second_centered_y(v))
    Y_n2_half += 0.5 * dt * nu * (der.second_centered_x(Y_n2) + der.second_centered_y(Y_n2))
    Y_o2_half += 0.5 * dt * nu * (der.second_centered_x(Y_o2) + der.second_centered_y(Y_o2))
    Y_ch4_half += 0.5 * dt * nu * (der.second_centered_x(Y_ch4) + der.second_centered_y(Y_ch4))
    Y_h2o_half += 0.5 * dt * nu * (der.second_centered_x(Y_h2o) + der.second_centered_y(Y_h2o))
    Y_co2_half += 0.5 * dt * nu * (der.second_centered_x(Y_co2) + der.second_centered_y(Y_co2))

    # Step 4: Compute reaction terms (source terms)
    concentration_ch4 = (rho / W_CH4) * Y_ch4_half
    concentration_o2 = (rho / W_O2) * Y_o2_half
    Q = A * concentration_ch4 * concentration_o2**2 * np.exp(-T_a / T)

    source_term_n2 = (nu_n2 / rho) * W_N2 * Q
    source_term_o2 = (nu_o2 / rho) * W_O2 * Q
    source_term_ch4 =( nu_ch4 / rho) * W_CH4 * Q
    source_term_h2o = (nu_h2o / rho) * W_H2O * Q
    source_term_co2 = (nu_co2 / rho) * W_CO2 * Q

    omega_T = - (
        (h_n2 / W_N2) * source_term_n2 +
        (h_o2 / W_O2) * source_term_o2 +
        (h_ch4 / W_CH4) * source_term_ch4 +
        (h_h2o / W_H2O) * source_term_h2o +
        (h_co2 / W_CO2) * source_term_co2
    )
    source_term_T = omega_T / (rho * c_p)

    # Step 5: Compute full-step values using half-step values and source terms
    u_star = u_half - dt * (u_half * der.derivative_x_centered(u_half) + v_half * der.derivative_y_centered(u_half))
    v_star = v_half - dt * (u_half * der.derivative_x_centered(v_half) + v_half * der.derivative_y_centered(v_half))
    Y_n2_star = Y_n2_half - dt * (u_half * der.derivative_x_centered(Y_n2_half) + v_half * der.derivative_y_centered(Y_n2_half)) + dt * source_term_n2
    Y_o2_star = Y_o2_half - dt * (u_half * der.derivative_x_centered(Y_o2_half) + v_half * der.derivative_y_centered(Y_o2_half)) + dt * source_term_o2
    Y_ch4_star = Y_ch4_half - dt * (u_half * der.derivative_x_centered(Y_ch4_half) + v_half * der.derivative_y_centered(Y_ch4_half)) + dt * source_term_ch4
    Y_h2o_star = Y_h2o_half - dt * (u_half * der.derivative_x_centered(Y_h2o_half) + v_half * der.derivative_y_centered(Y_h2o_half)) + dt * source_term_h2o
    Y_co2_star = Y_co2_half - dt * (u_half * der.derivative_x_centered(Y_co2_half) + v_half * der.derivative_y_centered(Y_co2_half)) + dt * source_term_co2
    T_star = T + dt * source_term_T

    # Add diffusion terms to full-step values
    u_star += dt * nu * (der.second_centered_x(u_half) + der.second_centered_y(u_half))
    v_star += dt * nu * (der.second_centered_x(v_half) + der.second_centered_y(v_half))
    Y_n2_star += dt * nu * (der.second_centered_x(Y_n2_half) + der.second_centered_y(Y_n2_half))
    Y_o2_star += dt * nu * (der.second_centered_x(Y_o2_half) + der.second_centered_y(Y_o2_half))
    Y_ch4_star += dt * nu * (der.second_centered_x(Y_ch4_half) + der.second_centered_y(Y_ch4_half))
    Y_h2o_star += dt * nu * (der.second_centered_x(Y_h2o_half) + der.second_centered_y(Y_h2o_half))
    Y_co2_star += dt * nu * (der.second_centered_x(Y_co2_half) + der.second_centered_y(Y_co2_half))

    # Step 6: Pressure correction (using SOR)
    P_new = sor(P, f=rho / dt * (der.derivative_x_centered(u_star) + der.derivative_y_centered(v_star)))

    # Step 7: Correct velocities using pressure gradient
    u_new = u_star - dt / rho * der.derivative_x_centered(P_new)
    v_new = v_star - dt / rho * der.derivative_y_centered(P_new)

    # Step 8: Apply boundary conditions to updated fields                                               
    u_new, v_new, P_new,T_star, Y_n2_star, Y_o2_star, Y_ch4_star = boundary_conditions(u_new, v_new, P_new,T_star, Y_n2_star, Y_o2_star, Y_ch4_star)

    return u_new, v_new, P_new,T_star, Y_n2_star, Y_o2_star, Y_ch4_star, Y_h2o_star, Y_co2_star

@jit(fastmath=True, nopython=True, cache=False)
def system_evolution_kernel_rk4(u:np.array, v:np.array, P:np.array, T:np.array, Y_n2:np.array, Y_o2:np.array, Y_ch4:np.array, Y_h2o:np.array, Y_co2:np.array):
    # Step 1
    u_star = u - dt * rk4_first_step_frac_u(u, v)
    v_star = v - dt * rk4_first_step_frac_v(u, v)

    # Boundary conditions
    u_star, v_star, P_star, T_star, Y_n2_star, Y_o2_star, Y_ch4_star = boundary_conditions(u_star, v_star, P_star, T, Y_n2_star, Y_o2_star, Y_ch4_star)

    # Species transport (RK4-based update)
    concentration_ch4 = rho / W_CH4 * Y_ch4
    concentration_o2 = rho / W_O2 * Y_o2
    Q = A * concentration_ch4 * concentration_o2**2 * np.exp(-T_a / T)

    source_term_n2 = (nu_n2 / rho) * W_N2 * Q
    source_term_o2 = (nu_o2 / rho) * W_O2 * Q
    source_term_ch4 = (nu_ch4 / rho) * W_CH4 * Q
    source_term_h2o = (nu_h2o / rho) * W_H2O * Q
    source_term_co2 = (nu_co2 / rho) * W_CO2 * Q

    # omega_T = - (h_n2 / W_N2 * rho * source_term_n2
    #              + h_o2 / W_O2 * rho * source_term_o2
    #              + h_ch4 / W_CH4 * rho * source_term_ch4
    #              + h_h2o / W_H2O * rho * source_term_h2o
    #              + h_co2 / W_CO2 * rho * source_term_co2)
    
    # source_term_T = omega_T / (rho * c_p)

    Y_n2_new = Y_n2 + dt * rk4_step_Y_k(Y_n2, -u, v, source_term_n2)
    Y_o2_new = Y_o2 + dt * rk4_step_Y_k(Y_o2, -u, v, source_term_o2)
    Y_ch4_new = Y_ch4 + dt * rk4_step_Y_k(Y_ch4, -u, v, source_term_ch4)
    Y_h2o_new = Y_h2o + dt * rk4_step_Y_k(Y_h2o, -u, v, source_term_h2o)
    Y_co2_new = Y_co2 + dt * rk4_step_Y_k(Y_co2, -u, v, source_term_co2)

    # T_new = T + dt * rk4_step_Y_k(T, -u, v, source_term_T) # The Y_k method can also be used for T!

    # Step 2
    u_double_star = u_star + dt * rk4_second_step_frac(u_star)
    v_double_star = v_star + dt * rk4_second_step_frac(v_star)

    # Step 3 (P is updated)
    P_new = sor(P, f=rho / dt * (der.derivative_x_centered(u_double_star) + der.derivative_y_centered(v_double_star)))

    # Step 4
    u_new = u_double_star - dt / rho * rk4_fourth_step_frac_u(P)
    v_new = v_double_star - dt / rho * rk4_fourth_step_frac_v(P)

    # Boundary conditions
    u_new, v_new, P_new, T_new, Y_n2_new, Y_o2_new, Y_ch4_new = boundary_conditions(u_new, v_new, P_new, T, Y_n2_new, Y_o2_new, Y_ch4_new)
        
    return u_new, v_new, P_new, T_new, Y_n2_new, Y_o2_new, Y_ch4_new, Y_h2o_new, Y_co2_new




#######################################
# Plotting the initial velocity field #
#######################################

u, v, P, T, Y_n2, Y_o2, Y_ch4 = boundary_conditions(u, v, P, T, Y_n2, Y_o2, Y_ch4)

plots.plot_vector_field(u, v, output_folder, dpi)


#############
# Main loop #
#############

# Lists to store velocity fields at different timesteps
u_history = []
v_history = []
P_history = []
T_history = []
strain_rate_history = []
Y_n2_history = []
Y_o2_history = []
Y_ch4_history = []
Y_h2o_history = []
Y_co2_history = []

for it in tqdm(range(nb_timesteps)):

    u_new, v_new, P_new, T_new, Y_n2_new, Y_o2_new, Y_ch4_new, Y_h2o_new, Y_co2_new = system_evolution_kernel_lw(u, v, P, T, Y_n2, Y_o2, Y_ch4, Y_h2o, Y_co2)

    if convergence_by_tolerance: 
        residual = np.linalg.norm(v - v_new, ord=2) # For example, with the v-field
        if np.linalg.norm(v, ord=2) > 10e-10:  # Avoid divide-by-zero by checking the norm
            residual /= np.linalg.norm(v, ord=2)

        if residual < tolerance_sys:
            break

    # Updating of the new fields
    u = np.copy(u_new)
    v = np.copy(v_new)
    P = np.copy(P_new)
    T = np.copy(T_new)
    Y_n2 = np.copy(Y_n2_new)
    Y_o2 = np.copy(Y_o2_new)
    Y_ch4 = np.copy(Y_ch4_new)
    Y_h2o = np.copy(Y_h2o_new)
    Y_co2 = np.copy(Y_co2_new)

    # Strain rate calculation
    strain_rate = np.abs(der.derivative_y_centered(v))
    max_strain_rate = np.max(strain_rate[:, 1])

    if it % 1 == 0:
        u_history.append(u.copy())
        v_history.append(v.copy())
        P_history.append(P.copy())
        T_history.append(T.copy())
        strain_rate_history.append(max_strain_rate.copy())
        Y_n2_history.append(Y_n2.copy())
        Y_o2_history.append(Y_o2.copy())
        Y_ch4_history.append(Y_ch4.copy())
        Y_h2o_history.append(Y_h2o.copy())
        Y_co2_history.append(Y_co2.copy())


u_history = np.array(u_history)
v_history = np.array(v_history)
P_history = np.array(P_history)
T_history = np.array(T_history)
strain_rate_history = np.array(strain_rate_history)
Y_n2_history = np.array(Y_n2_history)
Y_o2_history = np.array(Y_o2_history)
Y_ch4_history = np.array(Y_ch4_history)
Y_h2o_history = np.array(Y_h2o_history)
Y_co2_history = np.array(Y_co2_history)


#########################
# Strain rate over time #
#########################

plt.plot(strain_rate_history)
plt.title('Maximum strain rate on the left wall function of the number of iterations')
plt.yscale('log')
filename = 'Strain rate'
plt.savefig(os.path.join(output_folder, filename), dpi = dpi)

if show_figures: 
    plt.show()
else:
    plt.close()


########################
# Final pressure field #
########################

plt.figure(figsize=(6, 5))
plt.imshow(P, cmap='viridis')  # Display the data as an image
plt.colorbar(label='Value')  # Add a colorbar with a label
plt.title('Pressure field')  # Add a title
plt.xlabel('X-axis')  # Label for the x-axis
plt.ylabel('Y-axis')  # Label for the y-axis
filename = 'Pressure field'
plt.savefig(os.path.join(output_folder, filename), dpi = dpi)

if show_figures: 
    plt.show()
else:
    plt.close()

#####################################
# Final velocity fields (magnitude) #
#####################################

plots.plot_velocity_fields(-u_history[-1], v_history[-1], Lx, Ly, show_figures, dpi, 'Final velocity fields', output_folder)


#####################################
# Final velocity fields (direction) #
#####################################

x = np.linspace(0, Lx, nb_points)
y = np.linspace(Ly, 0, nb_points)
X, Y = np.meshgrid(x, y,indexing='xy')
plt.quiver(X, Y, -u_history[-1], v_history[-1], color="black")
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Flow Field")
filename = 'Final velocity fields (direction)'
plt.savefig(os.path.join(output_folder, filename), dpi = dpi)

if show_figures: 
    plt.show()
else:
    plt.close()

##############################
# Final species distribution #
##############################

plots.plot_field(Y_n2_history[-1], output_folder, show_figures, dpi, 'Y_N2 (steady)')
plots.plot_field(Y_o2_history[-1], output_folder, show_figures, dpi, 'Y_O2 (steady)')
plots.plot_field(Y_ch4_history[-1], output_folder, show_figures, dpi, 'Y_CH4 (steady)')
plots.plot_field(Y_h2o_history[-1], output_folder, show_figures, dpi, 'Y_H2O (steady)')
plots.plot_field(Y_co2_history[-1], output_folder, show_figures, dpi, 'Y_CO2 (steady)')


###########################
# Final temperature field #
###########################

plots.plot_field(T, output_folder, show_figures, dpi, 'Temperature field')


###################################
# Animation of the flow evolution #
###################################

if compute_animations: 
    plots.animate_flow_evolution(-u_history, v_history, Lx, Ly, nb_points, L_slot, L_coflow, output_folder, dpi)


#########################################################################################
# Animation of the temperature field, the pressure field and the five species over time #
#########################################################################################

if compute_animations:
    plots.animate_field_evolution(T_history, Lx, Ly, nb_points, L_slot, L_coflow, output_folder, dpi, 'Temperature')

    plots.animate_field_evolution(P_history, Lx, Ly, nb_points, L_slot, L_coflow, output_folder, dpi, 'Pressure')

    plots.animate_field_evolution(Y_n2_history, Lx, Ly, nb_points, L_slot, L_coflow, output_folder, dpi, 'Y_N2')

    plots.animate_field_evolution(Y_o2_history, Lx, Ly, nb_points, L_slot, L_coflow, output_folder, dpi, 'Y_O2')

    plots.animate_field_evolution(Y_ch4_history, Lx, Ly, nb_points, L_slot, L_coflow, output_folder, dpi, 'Y_CH4')

    plots.animate_field_evolution(Y_h2o_history, Lx, Ly, nb_points, L_slot, L_coflow, output_folder, dpi, 'Y_H2O')

    plots.animate_field_evolution(Y_co2_history, Lx, Ly, nb_points, L_slot, L_coflow, output_folder, dpi, 'Y_CO2')



#########################################################################
# (Question 3.1) Calculation of the maximum strain rate at steady state #
#########################################################################

strain_rate = np.abs(der.derivative_y_centered(v))
max_strain_rate = np.max(strain_rate[:, 1])

print(f"Maximum strain rate (|∂v/∂y|) on left wall: {max_strain_rate:.6f} s^-1")


#######################################################################################################
# (Question 3.2) Measure of the thickness of the N2 diffusion region in the left wall at steady state #
#######################################################################################################

closest_index_top = np.abs(Y_n2_history[-1][:, 1] - 0.9 * Y_n2_history[0][-1, 1]).argmin()
closest_index_bottom = np.abs(Y_n2_history[-1][:, 1] - 0.1 * Y_n2_history[0][-1, 1]).argmin()
length_diffusion_n2 = closest_index_bottom / nb_points * Lx - closest_index_top / nb_points * Lx

print(f"The length of the diffusion region for N2 in the left wall is: {length_diffusion_n2}")
