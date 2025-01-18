import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
import derivatives as der
import plotting as plots
from constants import (
    nb_points, dt, nb_timesteps, Lx, Ly, L_slot, L_coflow, U_slot, U_coflow, 
    T_slot, T_coflow, dx, dy, max_iter_sor, omega, tolerance_sor, tolerance_sys, 
    rho, nu, A, T_a, c_p, W_N2, W_O2, W_CH4, W_H2O, W_CO2, 
    nu_ch4, nu_o2, nu_n2, nu_h2o, nu_co2, h_n2, h_o2, h_ch4, h_h2o, h_co2
)


#################################################
# Input on image storage path and figure saving #
#################################################

# Path where the images will be stored (the name of the folder is specified at the end of the string)
output_folder  = 'C:\\Users\\danie\\Desktop\\Code results\\run_1' 
dpi = 300 # For storing the images with high quality
show_figures = False # If this variable is set to false, all the images are stored in the selected path and are not shown here

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

# Temperature field
T = np.zeros((nb_points, nb_points))

# Species fields
Y_n2 = np.zeros((nb_points, nb_points))
Y_o2 = np.zeros((nb_points, nb_points))
Y_ch4 = np.zeros((nb_points, nb_points))
Y_h2o = np.zeros((nb_points, nb_points)) # Should we put boundary conditions here?
Y_co2 = np.zeros((nb_points, nb_points))


##########################################
# Successive Overrelaxation (SOR) method #
##########################################

@jit(fastmath=True, nopython=True)
def sor(P, f, tolerance_sor=tolerance_sor, max_iter_sor=max_iter_sor, omega=omega):
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

    coef = 2 * (1 / dx**2 + 1 / dy**2)

    for _ in range(max_iter_sor):
        P_old = P.copy()
        laplacian_P = np.zeros_like(P)
        
        for i in range(1, nb_points - 1):
            for j in range(1, nb_points - 1):
                laplacian_P[i, j] = (P_old[i+1, j] + P[i-1, j]) / dy**2 + (P_old[i, j+1] + P[i, j-1]) / dx**2
                
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


#######################
# Boundary conditions #
#######################

@jit(fastmath=True, nopython=True)
def boundary_conditions(u, v, T, Y_n2, Y_o2, Y_ch4):
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

    Y_n2[-1, 1 : int(L_slot / Lx * nb_points) + 1] = 0.767 # Initial condition for the nitrogen in air (bottom slot)
    Y_n2[0, int(L_slot / Lx * nb_points) : int((L_slot + L_coflow) / Lx * nb_points)] = 1
    Y_n2[-1, int(L_slot / Lx * nb_points) : int((L_slot + L_coflow) / Lx * nb_points)] = 1

    Y_o2[-1, 1 : int(L_slot / Lx * nb_points) + 1] = 0.233 # Initial condition for the oxigen in air (bottom slot)

    Y_ch4[0, 1 : int(L_slot / Lx * nb_points) + 1] = 1

    return u, v, T, Y_n2, Y_o2, Y_ch4


#########################
# First step of the RK4 #
#########################

@jit(fastmath=True, nopython=True)
def compute_rhs_first_u(u : np.array, v : np.array):
    derivative_x = der.derivative_x_centered(u)
    derivative_y = der.derivative_y_centered(u)

    rhs = (
        u * derivative_x + v * derivative_y
    )
    
    return rhs


@jit(fastmath=True, nopython=True)
def compute_rhs_first_v(u : np.array, v : np.array):
    derivative_x = der.derivative_x_centered(v)
    derivative_y = der.derivative_y_centered(v)

    rhs = (
        u * derivative_x + v * derivative_y
    )
    
    return rhs


@jit(fastmath=True, nopython=True)
def rk4_first_step_frac_u(u : np.array, v : np.array, dt=dt):
    
    k1 = compute_rhs_first_u(u, v)
    k2 = compute_rhs_first_u(u + 0.5 * dt * k1, v)
    k3 = compute_rhs_first_u(u + 0.5 * dt * k2, v)
    k4 = compute_rhs_first_u(u + dt * k3, v)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


@jit(fastmath=True, nopython=True)
def rk4_first_step_frac_v(u : np.array, v : np.array, dt=dt):
    
    k1 = compute_rhs_first_v(u, v)
    k2 = compute_rhs_first_v(u, v + 0.5 * dt * k1)
    k3 = compute_rhs_first_v(u, v + 0.5 * dt * k2)
    k4 = compute_rhs_first_v(u, v + dt * k3)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


##########################
# Second step of the RK4 #
##########################

@jit(fastmath=True, nopython=True)
def compute_rhs_second(u : np.array):
    second_derivative_x = der.second_centered_x(u)
    second_derivative_y = der.second_centered_y(u)

    rhs = (
        nu * (second_derivative_x + second_derivative_y)
    )
    return rhs

@jit(fastmath=True, nopython=True)
def rk4_second_step_frac(u : np.array, dt=dt):
    
    k1 = compute_rhs_second(u)
    k2 = compute_rhs_second(u + 0.5 * dt * k1)
    k3 = compute_rhs_second(u + 0.5 * dt * k2)
    k4 = compute_rhs_second(u + dt * k3)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


#################################################
# Fourth step of the RK4 (the third one is SOR) #
#################################################

@jit(fastmath=True, nopython=True)
def compute_rhs_u(P : np.array):
    derivative_x = der.derivative_x_centered(P)

    rhs = (
        derivative_x
    )
    return rhs


@jit(fastmath=True, nopython=True)
def compute_rhs_v(P):
    derivative_y = der.derivative_y_centered(P)

    rhs = (
        derivative_y
    )
    return rhs


@jit(fastmath=True, nopython=True)
def rk4_fourth_step_frac_u(P : np.array, dt = dt):

    k1 = compute_rhs_u(P)
    k2 = compute_rhs_u(P + 0.5 * dt * k1)
    k3 = compute_rhs_u(P + 0.5 * dt * k2)
    k4 = compute_rhs_u(P + dt * k3)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


@jit(fastmath=True, nopython=True)
def rk4_fourth_step_frac_v(P : np.array, dt = dt):
    
    k1 = compute_rhs_v(P)
    k2 = compute_rhs_v(P + 0.5 * dt * k1)
    k3 = compute_rhs_v(P + 0.5 * dt * k2)
    k4 = compute_rhs_v(P + dt * k3)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


#######################################
# Solving species transport using RK4 #
#######################################

@jit(fastmath=True, nopython=True)
def compute_rhs_Y_k(Y_k, u : np.array, v : np.array, source_term: np.array):
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


@jit(fastmath=True, nopython=True)
def rk4_step_Y_k(Y_k, u : np. array, source_term: np.array, dt=dt):
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
def system_evolution_kernel_rk4(u, v, P, T, Y_n2, Y_o2, Y_ch4, Y_h2o, Y_co2):
    # Step 1
    u_star = u - dt * rk4_first_step_frac_u(u, v)
    v_star = v - dt * rk4_first_step_frac_v(u, v)

    # Species transport (RK4-based update)
    concentration_ch4 = rho / W_CH4 * Y_ch4
    concentration_o2 = rho / W_O2 * Y_o2
    Q = A * concentration_ch4 * concentration_o2**2 * np.exp(-T_a / T)

    source_term_n2 = nu_n2 / rho * W_N2 * Q
    source_term_o2 = nu_o2 / rho * W_O2 * Q
    source_term_ch4 = nu_ch4 / rho * W_CH4 * Q
    source_term_h2o = nu_h2o / rho * W_H2O * Q
    source_term_co2 = nu_co2 / rho * W_CO2 * Q

    omega_T = - (h_n2 / W_N2 * rho * source_term_n2
                 + h_o2 / W_O2 * rho * source_term_o2
                 + h_ch4 / W_CH4 * rho * source_term_ch4
                 + h_h2o / W_H2O * rho * source_term_h2o
                 + h_co2 / W_CO2 * rho * source_term_co2)
    
    source_term_T = omega_T / (rho * c_p)

    Y_n2_new = Y_n2 + dt * rk4_step_Y_k(Y_n2, u, v, source_term_n2)
    Y_o2_new = Y_o2 + dt * rk4_step_Y_k(Y_o2, u, v, source_term_o2)
    Y_ch4_new = Y_ch4 + dt * rk4_step_Y_k(Y_ch4, u, v, source_term_ch4)
    Y_h2o_new = Y_h2o + dt * rk4_step_Y_k(Y_h2o, u, v, source_term_h2o)
    Y_co2_new = Y_co2 + dt * rk4_step_Y_k(Y_co2, u, v, source_term_co2)

    T_new = T + dt * rk4_step_Y_k(T, u, v, source_term_T) # The Y_k method can also be used for T!

    # Step 2
    u_double_star = u_star + dt * rk4_second_step_frac(u_star)
    v_double_star = v_star + dt * rk4_second_step_frac(v_star)

    # Step 3 (P is updated)
    P_new = sor(P, f=rho / dt * (der.derivative_x_centered(u_double_star) + der.derivative_y_centered(v_double_star)))

    # Step 4
    u_new = u_double_star - dt / rho * rk4_fourth_step_frac_u(P)
    v_new = v_double_star - dt / rho * rk4_fourth_step_frac_v(P)

    # Boundary conditions
    u_new, v_new, T_new, Y_n2_new, Y_o2_new, Y_ch4_new = boundary_conditions(u_new, v_new, T_new, Y_n2_new, Y_o2_new, Y_ch4_new)

    P[:, 0] = 0
    P[:, 1] = P[:, 2] # dP/dx = 0 at the left wall
    P[0, 1:] = P[1, 1:] # dP/dy = 0 at the top wall
    P[-1, 1:] = P[-2, 1:] # dP/dy = 0 at the bottom wall
    P[:, -1] = 0 # P = 0 at the right free limit
        
    return u_new, v_new, P_new, T_new, Y_n2_new, Y_o2_new, Y_ch4_new, Y_h2o_new, Y_co2_new




#######################################
# Plotting the initial velocity field #
#######################################

u, v, T, Y_n2, Y_o2, Y_ch4 = boundary_conditions(u, v, T, Y_n2, Y_o2, Y_ch4)

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

    u_new, v_new, P_new, T_new, Y_n2_new, Y_o2_new, Y_ch4_new, Y_h2o_new, Y_co2_new = system_evolution_kernel_rk4(u, v, P, T, Y_n2, Y_o2, Y_ch4, Y_h2o, Y_co2)

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
plt.imshow(-P, cmap='viridis')  # Display the data as an image
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

plots.animate_flow_evolution(-u_history, v_history, Lx, Ly, nb_points, L_slot, L_coflow, output_folder, dpi)


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
