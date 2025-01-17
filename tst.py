import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit

# To correctly import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules import plot
from modules.derivatives import *
from modules.constants import *
from modules import schemes

dx = Lx / (nb_points - 1)
dy = Ly / (nb_points - 1)

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

# Lists to store velocity fields at different timesteps
u_history = []
v_history = []
strain_rate_history = []
Y_n2_history = []

u, v, Y_n2 = boundary_conditions(u, v, Y_n2)

# Plotting the initial velocity field
u, v, Y_n2 = boundary_conditions(u, v, Y_n2)
fig, ax = plt.subplots(figsize=(6, 5))
plot.streamplot(ax,u,v,Lx,Ly,nb_points)
plt.show()

for it in tqdm(range(nb_timesteps)):

    u_new, v_new, P, Y_n2_double_star = schemes.lax_wendroff(u, v, P, Y_n2)

    #another criterium
    residual=v_new - v
    if np.linalg.norm(residual, ord=np.inf)< tol_sys:
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


fig, ax = plt.subplots(figsize=(6, 5))

# Plot pressure
pressure_plot = plot.P(ax, P, Lx, Ly,vmin=abs(P).min(),vmax=abs(P).max())

# Add streamplot on the same axis
plot.streamplot(ax, u_history[-1], v_history[-1], Lx, Ly, nb_points)

# Add colorbar for the pressure
plt.colorbar(pressure_plot, ax=ax, label="Pressure")

# Show the plot
plt.show()

#plot.velocity_fields(-u_history[-1], v_history[-1], Lx, Ly, nb_points, L_slot, L_coflow)
#plot.animate_flow_evolution(u_history, v_history, Lx, Ly, nb_points, L_slot, L_coflow)

closest_index_top = np.abs(Y_n2_history[-1][:, 0] - 0.9 * Y_n2_history[0][-1, 0]).argmin()
closest_index_bottom = np.abs(Y_n2_history[-1][:, 0] - 0.1 * Y_n2_history[0][-1, 0]).argmin()
length_diffusion = -closest_index_top / nb_points * Lx + closest_index_bottom / nb_points * Lx

# Once the u field is correctly advected, we calculate "a"
strain_rate = np.abs(derivative_y_centered(v))

max_strain_rate = np.max(strain_rate[:, 1])

print(f"Maximum strain rate (|∂v/∂y|) on left wall: {max_strain_rate:.6f} s^-1")
