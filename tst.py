import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit


# We had a problem with the path of the other module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules import plot
from modules.derivatives import *
#############
# Constants #
#############

nb_points = 42

R = 8.314  # Ideal gas constant in J/(mol* K)

M_N2 = 0.02802  # kg/mol

M_air = 0.02895  # kg/mol

M_CH4 = 0.016042  # kg/mol

dt = 1e-6

final_time = 1

nb_timesteps = int(final_time / dt)

Lx = 2e-3

Ly = 2e-3

L_slot = 0.5e-3

L_coflow = 0.5e-3

U_slot = 1.0

T_slot = 300

U_coflow = 0.2

T_coflow = 300

rho = 1.1614 # Fluid density

dx = Lx / (nb_points - 1)

dy = Ly / (nb_points - 1)

nu = 15e-6

tolerance_sor = 1e-7 # Tolerance for the convergence of the SOR algorithm

tolerance_sys = 1e-4 # Tolerance for the convergence of the whole system

max_iter = 10000  # Maximum number of iterations for achieving convergence

omega = 1.5  # Parameter for the Successive Overrelaxation Method (SOR), it has to be between 1 and 2


##################
# Initial fields #
##################

# Velocity fields
v = np.zeros((nb_points, nb_points))
u = np.zeros((nb_points, nb_points))

# Pressure field
P = np.zeros((nb_points, nb_points))

# Species field (nitrogen)
Y_n2 = np.zeros((nb_points, nb_points))


#################################################
# Derivatives and second derivatives definition #
#################################################

@jit(fastmath=True, nopython=True)
def sor(P, f, tolerance_sor=tolerance_sor, max_iter=max_iter, omega=omega):
    """
    Successive Overrelaxation (SOR) method for solving the pressure Poisson equation.
    Optimized using Numba

    Parameters:
        P (np.array): Initial guess for the pressure field.
        f (np.array): Right-hand side of the Poisson equation.
        tolerance (float): Convergence tolerance for the iterative method.
        max_iter (int): Maximum number of iterations.
        omega (float): Relaxation factor, between 1 and 2.

    Returns:
        np.array: Updated pressure field.
    """

    coef = 2 * (1 / dx**2 + 1 / dy**2)

    for _ in range(max_iter):
        P_old = P.copy()
        laplacian_P = np.zeros_like(P)
        
        for i in range(1, nb_points - 1):
            for j in range(1, nb_points - 1):
                laplacian_P[i, j] = (P[i+1, j] + P[i-1, j]) / dx**2 + (P[i, j+1] + P[i, j-1]) / dy**2
                
                # Update P using the SOR formula
                P[i, j] = (1 - omega) * P[i, j] + (omega / coef) * (laplacian_P[i, j] - f[i, j])
        
        # Compute the residual to check for convergence
        # residual = np.linalg.norm(P - P_old, ord=2)
        residual= P-P_old
        # if np.linalg.norm(residual, ord=np.inf) > 10e-10:  # Avoid divide-by-zero by checking the norm
        #     residual /= np.linalg.norm(P_old, ord=2)

        if np.linalg.norm(residual, ord=np.inf) < tolerance_sor:
            return P
    print("SOR did not converge")
    return P


##########################
# Fractional-step method #
##########################

# Lists to store velocity fields at different timesteps
u_history = []
v_history = []
strain_rate_history = []
Y_n2_history = []

@jit(fastmath=True, nopython=True)
def boundary_conditions(u, v, Y_n2):
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

    Y_n2[-1, 1 : int(L_slot / Lx * nb_points) + 1] = 0.767 # Initial condition for the nitrogen in air (bottom slot)
    Y_n2[0, int(L_slot / Lx * nb_points) : int((L_slot + L_coflow) / Lx * nb_points)] = 1
    Y_n2[-1, int(L_slot / Lx * nb_points) : int((L_slot + L_coflow) / Lx * nb_points)] = 1

    return u, v, Y_n2

u, v, Y_n2 = boundary_conditions(u, v, Y_n2)

@jit(fastmath=True, nopython=True)
def system_evolution_kernel(u, v, P, Y_n2):
    
    #u, v, Y_n2 = boundary_conditions(u, v, Y_n2)

    # Step 1
    u_star = u - dt * (u * derivative_x_upwind(u,u) + v * derivative_y_upwind(u,v)) 
    v_star = v - dt * (u * derivative_x_upwind(v,u) + v * derivative_y_upwind(v,v)) 
    
    # Species transport 
    Y_n2_star = Y_n2 - dt * (u * derivative_x_centered(Y_n2) + v * derivative_y_centered(Y_n2))

    u_star, v_star, Y_n2_star = boundary_conditions(u_star, v_star, Y_n2_star)

    # Step 2
    u_double_star = u_star + dt * (nu * (second_centered_x(u_star) + second_centered_y(u_star)))
    v_double_star = v_star + dt * (nu * (second_centered_x(v_star) + second_centered_y(v_star)))
    Y_n2_double_star = Y_n2_star + dt * (nu * (second_centered_x(Y_n2_star) + second_centered_y(Y_n2_star)))

    u_double_star, v_double_star, Y_n2_double_star = boundary_conditions(u_double_star, v_double_star, Y_n2_double_star)

    P[:, 0] = 0
    P[:, 1] = P[:, 2] # dP/dx = 0 at the left wall
    P[0, 1:] = P[1, 1:] # dP/dy = 0 at the top wall
    P[-1, 1:] = P[-2, 1:] # dP/dy = 0 at the bottom wall
    P[:, -1] = 0 # P = 0 at the right free limit

    # Step 3 (P is updated)
    P = sor(P, f=rho / dt * (derivative_x_centered(u_double_star) + derivative_y_centered(v_double_star)))

    P[:, 0] = 0
    P[:, 1] = P[:, 2] # dP/dx = 0 at the left wall
    P[0, 1:] = P[1, 1:] # dP/dy = 0 at the top wall
    P[-1, 1:] = P[-2, 1:] # dP/dy = 0 at the bottom wall
    P[:, -1] = 0 # P = 0 at the right free limit
   
    # Step 4
    u_new = u_double_star - dt / rho * derivative_x_upwind(P, np.ones_like(P))
    v_new = v_double_star - dt / rho * derivative_y_upwind(P, np.ones_like(P))

    u_new, v_new, Y_n2_double_star = boundary_conditions(u_new, v_new, Y_n2_double_star)

    # P[:, 0] = 0
    # P[:, 1] = P[:, 2] # dP/dx = 0 at the left wall
    # P[0, 1:] = P[1, 1:] # dP/dy = 0 at the top wall
    # P[-1, 1:] = P[-2, 1:] # dP/dy = 0 at the bottom wall
    # P[:, -1] = 0 # P = 0 at the right free limit

    return u_new, v_new, P, Y_n2_double_star


# Plotting the initial velocity field
u, v, Y_n2 = boundary_conditions(u, v, Y_n2)
plot.streamplot(u,v,Lx,Ly,nb_points)
# plt.quiver(X*1e3,Y*1e3,u,v,scale=10)
# plt.title('Initial velocity field configuration')
# plt.xlabel('Lx (mm)')
# plt.ylabel('Ly (mm)')

plt.show()


for it in tqdm(range(nb_timesteps)):

    u_new, v_new, P, Y_n2_double_star = system_evolution_kernel(u, v, P, Y_n2)

    residual = np.linalg.norm(v - v_new, ord=2)
    if np.linalg.norm(v, ord=2) > 10e-10:  # Avoid divide-by-zero by checking the norm
        residual /= np.linalg.norm(v, ord=2)

    if residual < tolerance_sys:
        break

    # Updating of the new fields
    u = np.copy(u_new)
    v = np.copy(v_new)
    Y_n2 = np.copy(Y_n2_double_star)

    # Strain rate
    strain_rate = np.abs(derivative_y_centered(v))
    max_strain_rate = np.max(strain_rate[:, 1])


    if it % 1 == 0:
        u_history.append(u.copy())
        v_history.append(v.copy())
        strain_rate_history.append(max_strain_rate)
        Y_n2_history.append(Y_n2)

plt.plot(strain_rate_history)
plt.title('Maximum strain rate on the left wall function of the number of iterations')
plt.yscale('log')
plt.show()

plt.figure(figsize=(6, 5))
plot.P(P)
# Show the plot



#plot.velocity_fields(-u_history[-1], v_history[-1], Lx, Ly, nb_points, L_slot, L_coflow)

# Vector velocity field
plot.streamplot(u_history[-1],v_history[-1],Lx,Ly,nb_points)
plt.show()

#plot.animate_flow_evolution(u_history, v_history, Lx, Ly, nb_points, L_slot, L_coflow)

closest_index_top = np.abs(Y_n2_history[-1][:, 0] - 0.9 * Y_n2_history[0][-1, 0]).argmin()
closest_index_bottom = np.abs(Y_n2_history[-1][:, 0] - 0.1 * Y_n2_history[0][-1, 0]).argmin()
length_diffusion = -closest_index_top / nb_points * Lx + closest_index_bottom / nb_points * Lx

# Once the u field is correctly advected, we calculate "a"
strain_rate = np.abs(derivative_y_centered(v))

max_strain_rate = np.max(strain_rate[:, 1])

print(f"Maximum strain rate (|∂v/∂y|) on left wall: {max_strain_rate:.6f} s^-1")
