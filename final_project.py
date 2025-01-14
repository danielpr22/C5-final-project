import sys
import os
import string
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
import numpy as np


# We had a problem with the path of the other module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#############
# Constants #
#############

nb_points = 50

R = 8.314  # Ideal gas constant in J/(mol* K)

M_N2 = 0.02802  # kg/mol

M_air = 0.02895  # kg/mol

M_CH4 = 0.016042  # kg/mol

dt = 1e-6

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

dx = Lx / (nb_points - 1)

dy = Ly / (nb_points - 1)

nu = 15e-6

tolerance = 1e-4

max_iter = 10000  # Maximum number of iterations

omega = 1.88  # Parameter for the Successive Overrelaxation Method (SOR), it has to be between 1 and 2


##################
# Initial fields #
##################

# Velocity fields
v = np.zeros((nb_points, nb_points))
u = np.zeros((nb_points, nb_points))

# Pressure field
P = np.zeros((nb_points, nb_points))

u[:, 0] = 0
u[0, :] = 0
u[-1, :] = 0
v[0, 0 : int(L_slot / Lx * nb_points)] = -U_slot # Speed at the top of the "flow" region (Methane)
v[-1, 0 : int(L_slot / Lx * nb_points)] = U_slot  # Speed at the bottom of the "flow" region (Air)
v[0, int(L_slot / Lx * nb_points) : int((L_slot + L_coflow) / Lx * nb_points)] = -U_coflow  # Speed at the top of the "coflow" region (Nitrogen)
v[-1, int(L_slot / Lx * nb_points) : int((L_slot + L_coflow) / Lx * nb_points)] = U_coflow

# Plotting an initial field
# plt.imshow(np.where(u != np.nan, u, np.nan), origin="lower")
# plt.colorbar()
# plt.show()

#################################################
# Derivatives and second derivatives definition #
#################################################

def derivative_x(u: np.array, var_name: string, dx=dx) -> np.array:
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

    u_left = np.roll(u, shift=1, axis=0)  # u[i-1, j]
    u_right = np.roll(u, shift=-1, axis=0)  # u[i+1, j]

    # Apply upwind scheme based on the sign of u

    derivative = np.where(u > 0, (u - u_left) / dx, (u_right - u) / dx)

    if var_name == "u":
        derivative[:, -1] = 0  # Right free limit condition

    elif var_name == "v":
        derivative[:, 0] = 0  # Left slippery wall
        derivative[:, -1] = 0  # Right free limit condition

    elif var_name == "P":
        derivative[:, 0] = 0  # Left slippery wall

    return derivative

def derivative_y(u: np.array, var_name: string, dy=dy) -> np.array:
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
    u_left = np.roll(u, shift=1, axis=1)  # u[i, j - 1]
    u_right = np.roll(u, shift=-1, axis=1)  # u[i, j + 1]

    # Apply upwind scheme based on the sign of u
    derivative = np.where(u > 0, (u - u_left) / dy, (u_right - u) / dy)

    if var_name == "P":
        derivative[0, :] = 0
        derivative[-1, :] = 0

    return derivative

def second_centered_x(u: np.array, dx=dx) -> np.array:

    u_left = np.roll(u, shift=1, axis=0)  # u[i-1, j]
    u_right = np.roll(u, shift=-1, axis=0)  # u[i+1, j]

    return (u_right - 2 * u + u_left) / dx**2

def second_centered_y(u: np.array, dy=dy) -> np.array:

    u_left = np.roll(u, shift=1, axis=1)  # u[i, j - 1]
    u_right = np.roll(u, shift=-1, axis=1)  # u[i, j + 1]

    return (u_right - 2 * u + u_left) / dy**2


# @jit(nopython=True, fastmath=True, parallel=True, cache=False)
# def sor_kernel(P, f, coef, omega, dx2_inv, dy2_inv, nb_points):
#     """Optimized SOR kernel using Numba"""
#     P_old = np.copy(P)
#     P_new = np.copy(P)
#     P_aux= np.copy(P)
#     for i in range(1, nb_points - 1):  # Exclude boundary points
#         for j in range(1, nb_points - 1):
#             # Compute the new pressure using the SOR formula
#             P_aux[i, j] = coef * ((P_old[i + 1, j] + P_new[i - 1, j]) * dx2_inv
#                             + (P_old[i, j + 1] + P_new[i, j - 1]) * dy2_inv - f[i, j])
            
#             P_new[i, j] = (1 - omega) * P_old[i, j] + omega * P_aux
#             P_old = np.copy(P_new)

#     return P


# def sor(P: np.array, f: np.array, dx=dx, dy=dy, tol=tolerance, max_iter=max_iter) -> np.array:
#     """
#     Solve the Poisson equation for pressure correction using Successive Over-Relaxation (SOR).

#     Parameters:
#         P (np.array): Initial pressure field (2D array).
#         f (np.array): Source term (right-hand side of the Poisson equation).
#         omega (float): Relaxation factor for SOR (1 < omega < 2).
#         tol (float): Convergence tolerance.
#         max_iter (int): Maximum number of iterations.

#     Returns:
#         np.array: Updated pressure field.
#     """
#     dx2_inv = 1 / (dx**2)
#     dy2_inv = 1 / (dy**2)
#     coef = 1 / (2 * (dx2_inv + dy2_inv))
#     # Coefficient for the discretized Poisson equation

#     for it in tqdm(range(max_iter)):
#         P_old = P.copy()  # Save the old pressure field for convergence check

#         # Update pressure using SOR
#         # P_new = np.zeros_like(P)

#         P = sor_kernel(P, f, dx2_inv=dx2_inv, dy2_inv=dy2_inv, omega=omega, nb_points=nb_points, coef=coef)
#         residual = P - P_old

#         # Check for convergence using the infinity norm of the difference (maximum difference amongst all the points)
#         if (np.linalg.norm(residual, ord=np.inf) <= tol):
#             print(f"SOR converged after {it+1} iterations.")
#             return P

#     print("SOR did not converge within the maximum number of iterations.")
#     return P


def sor(P, f, tolerance=tolerance, max_iter=max_iter, omega=omega):
    """
    Successive Overrelaxation (SOR) method for solving the pressure Poisson equation.

    Parameters:
        P (np.array): Initial guess for the pressure field.
        f (np.array): Right-hand side of the Poisson equation.
        tolerance (float): Convergence tolerance for the iterative method.
        max_iter (int): Maximum number of iterations.
        omega (float): Relaxation factor, between 1 and 2.

    Returns:
        np.array: Updated pressure field.
    """
    # Grid dimensions
    ny, nx = P.shape
    
    for iteration in range(max_iter):
        P_old = P.copy()
        
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                # Discretized Laplacian of P
                laplacian_P = (P[i+1, j] + P[i-1, j]) / dx**2 + (P[i, j+1] + P[i, j-1]) / dy**2
                coefficient = 2 * (1 / dx**2 + 1 / dy**2)
                
                # Update P using the SOR formula
                P[i, j] = (1 - omega) * P[i, j] + (omega / coefficient) * (f[i, j] - laplacian_P)
        
        # Compute the residual to check for convergence
        residual = np.linalg.norm(P - P_old, ord=2)
        if np.linalg.norm(P_old, ord=2) > 1e-10:  # Avoid divide-by-zero by checking the norm
            residual /= np.linalg.norm(P_old, ord=2)
        if residual < tolerance:
            print(f"SOR converged after {iteration + 1} iterations with residual {residual:.2e}.")
            break
    else:
        print(f"SOR did not converge after {max_iter} iterations. Residual: {residual:.2e}")

    return P


##########################
# Fractional-step method #
##########################

def plot_velocity_fields(u, v, Lx, Ly, nb_points, L_slot, L_coflow, save_path=None):
    """
    Plot the velocity fields u and v using colormaps, accounting for matrix indexing.

    Parameters:
        u (np.array): x-component of velocity field
        v (np.array): y-component of velocity field
        Lx (float): Domain length in x direction
        Ly (float): Domain length in y direction
        nb_points (int): Number of grid points
        L_slot (float): Length of the slot
        L_coflow (float): Length of the coflow region
        save_path (str, optional): Path to save the plots
    """
    # Create coordinate meshgrid
    x = np.linspace(0, Lx, nb_points)
    y = np.linspace(Ly, 0, nb_points)  # Reversed for matrix indexing
    X, Y = np.meshgrid(x, y)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot u velocity field
    c1 = ax1.pcolormesh(X, Y, u, cmap="RdBu_r", shading="auto")
    ax1.set_title("u-velocity field")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    plt.colorbar(c1, ax=ax1, label="Velocity (m/s)")

    # Add lines to show slot and coflow regions
    ax1.axvline(x=L_slot, color="k", linestyle="--", alpha=0.5)
    ax1.axvline(x=L_slot + L_coflow, color="k", linestyle="--", alpha=0.5)

    # Plot v velocity field
    c2 = ax2.pcolormesh(X, Y, v, cmap="RdBu_r", shading="auto")
    ax2.set_title("v-velocity field")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    plt.colorbar(c2, ax=ax2, label="Velocity (m/s)")

    # Add lines to show slot and coflow regions
    ax2.axvline(x=L_slot, color="k", linestyle="--", alpha=0.5)
    ax2.axvline(x=L_slot + L_coflow, color="k", linestyle="--", alpha=0.5)

    # Add text annotations for the regions
    def add_region_labels(ax):
        ax.text(L_slot / 2, Ly - 0.1e-3, "Slot", ha="center")
        ax.text(L_slot + L_coflow / 2, Ly - 0.1e-3, "Coflow", ha="center")
        ax.text(
            L_slot + L_coflow + (Lx - L_slot - L_coflow) / 2,
            Ly - 0.1e-3,
            "External",
            ha="center",
        )

    add_region_labels(ax1)
    add_region_labels(ax2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def animate_flow_evolution(
    u_history,
    v_history,
    Lx,
    Ly,
    nb_points,
    L_slot,
    L_coflow,
    interval=1500,
    skip_frames=2,
    save_path='C:\\Users\\danie\\Desktop\\animation.gif',
):
    """
    Create a slower animation of the flow evolution over time, accounting for matrix indexing.

    Parameters:
        u_history (list): List of u velocity fields at different timesteps
        v_history (list): List of v velocity fields at different timesteps
        Lx (float): Domain length in x direction
        Ly (float): Domain length in y direction
        nb_points (int): Number of grid points
        L_slot (float): Length of the slot
        L_coflow (float): Length of the coflow region
        interval (int): Interval between frames in milliseconds (default: 200ms for slower animation)
        skip_frames (int): Number of frames to skip between each animation frame (default: 2)
        save_path (str, optional): Path to save the animation
    """
    from matplotlib.animation import FuncAnimation

    # Create coordinate meshgrid with reversed y-axis for matrix indexing
    x = np.linspace(0, Lx, nb_points)
    y = np.linspace(Ly, 0, nb_points)  # Reversed for matrix indexing
    X, Y = np.meshgrid(x, y)

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Calculate value ranges for consistent colorbars
    u_min = min(np.min(u) for u in u_history)
    u_max = max(np.max(u) for u in u_history)
    v_min = min(np.min(v) for v in v_history)
    v_max = max(np.max(v) for v in v_history)

    # Initialize plots with consistent color scales
    c1 = ax1.pcolormesh(
        X, Y, u_history[0], cmap="RdBu_r", shading="auto", vmin=u_min, vmax=u_max
    )
    ax1.set_title("u-velocity field")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    plt.colorbar(c1, ax=ax1, label="Velocity (m/s)")

    # Add reference lines and labels for regions
    ax1.axvline(x=L_slot, color="k", linestyle="--", alpha=0.5)
    ax1.axvline(x=L_slot + L_coflow, color="k", linestyle="--", alpha=0.5)
    ax1.text(L_slot / 2, Ly - 0.1e-3, "Slot", ha="center")
    ax1.text(L_slot + L_coflow / 2, Ly - 0.1e-3, "Coflow", ha="center")
    ax1.text(
        L_slot + L_coflow + (Lx - L_slot - L_coflow) / 2,
        Ly - 0.1e-3,
        "External",
        ha="center",
    )

    c2 = ax2.pcolormesh(
        X, Y, v_history[0], cmap="RdBu_r", shading="auto", vmin=v_min, vmax=v_max
    )
    ax2.set_title("v-velocity field")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    plt.colorbar(c2, ax=ax2, label="Velocity (m/s)")

    # Add reference lines and labels for regions
    ax2.axvline(x=L_slot, color="k", linestyle="--", alpha=0.5)
    ax2.axvline(x=L_slot + L_coflow, color="k", linestyle="--", alpha=0.5)
    ax2.text(L_slot / 2, Ly - 0.1e-3, "Slot", ha="center")
    ax2.text(L_slot + L_coflow / 2, Ly - 0.1e-3, "Coflow", ha="center")
    ax2.text(
        L_slot + L_coflow + (Lx - L_slot - L_coflow) / 2,
        Ly - 0.1e-3,
        "External",
        ha="center",
    )

    # Add timestamp text
    timestamp = ax1.text(
        0.02, 1.02, f"Frame: 0/{len(u_history)}", transform=ax1.transAxes
    )

    # Take every nth frame for smoother, slower animation
    frame_indices = range(0, len(u_history), skip_frames)

    def update(frame_idx):
        frame = frame_indices[frame_idx]
        c1.set_array(u_history[frame].ravel())
        c2.set_array(v_history[frame].ravel())
        timestamp.set_text(f"Frame: {frame}/{len(u_history)}")
        return c1, c2, timestamp

    plt.tight_layout()

    anim = FuncAnimation(
        fig, update, frames=len(frame_indices), interval=interval, blit=True
    )

    if save_path:
        anim.save(save_path, writer="pillow")

    plt.show()


# Lists to store velocity fields at different timesteps
u_history = []
v_history = []
for it in tqdm(range(100)):

    # Speed at the top of the "coflow" region (Nitrogen)

    # Step 1
    u_star = u - dt * (u * derivative_x(u, "u") + v * derivative_y(u, "u"))
    v_star = v - dt * (u * derivative_x(v, "v") + v * derivative_y(v, "v"))

    u_star[:, 0] = 0
    u_star[0, :] = 0
    u_star[-1, :] = 0
    # Boundary conditions for u_star and v_star
    v_star[0, 0 : int(L_slot / Lx * nb_points)] = -U_slot  # Speed at the top of the "flow" region (Methane)
    v_star[-1, 0 : int(L_slot / Lx * nb_points)] = U_slot  # Speed at the bottom of the "flow" region (Air)
    v_star[0, int(L_slot / Lx * nb_points) : int((L_slot + L_coflow) / Lx * nb_points)] = -U_coflow  # Speed at the top of the "coflow" region (Nitrogen)
    v_star[-1, int(L_slot / Lx * nb_points) : int((L_slot + L_coflow) / Lx * nb_points)] = U_coflow  # Speed at the top of the "coflow" region (Nitrogen)

    # Step 2
    u_double_star = u_star + dt * (nu * (second_centered_x(u_star) + second_centered_y(u_star)))
    v_double_star = v_star + dt * (nu * (second_centered_x(v_star) + second_centered_y(v_star)))


    # Boundary conditions for P, u_double_star, v_double_star, d(u_double_star)/dx and d(v_double_star)/dy (the latter two are directly included in the derivative functions)
    u_double_star[:, 0] = 0
    u_double_star[0, :] = 0
    u_double_star[-1, :] = 0
    v_double_star[0, 0 : int(L_slot / Lx * nb_points)] = -U_slot  # Speed at the top of the "flow" region (Methane)
    v_double_star[-1, 0 : int(L_slot / Lx * nb_points)] = U_slot  # Speed at the bottom of the "flow" region (Air)
    v_double_star[0, int(L_slot / Lx * nb_points) : int((L_slot + L_coflow) / Lx * nb_points)] = -U_coflow  # Speed at the top of the "coflow" region (Nitrogen)
    v_double_star[-1, int(L_slot / Lx * nb_points) : int((L_slot + L_coflow) / Lx * nb_points)] = U_coflow  # Speed at the top of the "coflow" region (Nitrogen)

    # Step 3
    P = sor(P, f=rho / dt * (derivative_x(u_double_star, "u") + derivative_y(v_double_star, "v")))
   
    # Step 4
    u_new = u_double_star - dt / rho * derivative_x(P, "P")
    v_new = v_double_star - dt / rho * derivative_y(P, "P")

    u_new[:, 0] = 0
    u_new[0, :] = 0
    u_new[-1, :] = 0
    # Boundary conditions for P, u_new, v_new, d(u_new)/dx and d(v_new)/dy (the latter two are directly included in the derivative functions)
    v_new[0, 0 : int(L_slot / Lx * nb_points)] = -U_slot  # Speed at the top of the "flow" region (Methane)
    v_new[-1, 0 : int(L_slot / Lx * nb_points)] = U_slot  # Speed at the bottom of the "flow" region (Air)
    v_new[0, int(L_slot / Lx * nb_points) : int((L_slot + L_coflow) / Lx * nb_points)] = -U_coflow  # Speed at the top of the "coflow" region (Nitrogen)
    v_new[-1, int(L_slot / Lx * nb_points) : int((L_slot + L_coflow) / Lx * nb_points)] = U_coflow

    # Updating of the new fields
    u = np.copy(u_new)
    v = np.copy(v_new)

    if it % 1 == 0:
        u_history.append(u.copy())
        v_history.append(v.copy())


plot_velocity_fields(u_history[0], v_history[0], Lx, Ly, nb_points, L_slot, L_coflow)

animate_flow_evolution(u_history, v_history, Lx, Ly, nb_points, L_slot, L_coflow)


# Once the u field is correctly advected, we calculate "a"
strain_rate = np.abs(-derivative_x(u, "u"))  # Since dv/dy = -du/dx

max_strain_rate = np.max(strain_rate)

print(f"Maximum strain rate (|∂v/∂y|) on left wall: {max_strain_rate:.6f} s^-1")
