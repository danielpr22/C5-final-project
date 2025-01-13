import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm

# Adjust module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#############
# Constants #
#############
nb_points = 100
R = 8314 # Ideal gas constant in J/(kg * K)
dt = 1e-2
Lx = 2e-3
Ly = 2e-3
L_slot = 0.5e-3
L_coflow = 0.5e-3
U_slot = 1.0
T_slot = 300
U_coflow = 0.2
T_coflow = 300
rho = 1.1614 # Fluid density
nu = 15e-6  # Kinematic viscosity
dx = Lx / (nb_points - 1)
dy = Ly / (nb_points - 1)
tolerance = 1
max_iter = 20000
omega = 1.5

##################
# Initial fields #
##################
# Velocity fields
v = np.zeros((nb_points, nb_points))
u = np.zeros((nb_points, nb_points))
# Pressure field
P = np.zeros((nb_points, nb_points))

###########################
# Vectorized Laplacian #
###########################
def laplacian(field, dx=dx, dy=dy):
    """Compute the Laplacian of a field, explicitly handling boundaries."""
    lap = np.zeros_like(field)

    # Interior points
    lap[1:-1, 1:-1] = (
        (field[2:, 1:-1] + field[:-2, 1:-1] - 2 * field[1:-1, 1:-1]) / dx**2 +
        (field[1:-1, 2:] + field[1:-1, :-2] - 2 * field[1:-1, 1:-1]) / dy**2
    )

    # Top boundary
    lap[0, 1:-1] = (
        (field[1, 1:-1] - 2 * field[0, 1:-1] + field[1, 1:-1]) / dx**2 +
        (field[0, 2:] + field[0, :-2] - 2 * field[0, 1:-1]) / dy**2
    )

    # Bottom boundary
    lap[-1, 1:-1] = (
        (field[-2, 1:-1] - 2 * field[-1, 1:-1] + field[-2, 1:-1]) / dx**2 +
        (field[-1, 2:] + field[-1, :-2] - 2 * field[-1, 1:-1]) / dy**2
    )

    # Left boundary
    lap[1:-1, 0] = (
        (field[2:, 0] + field[:-2, 0] - 2 * field[1:-1, 0]) / dx**2 +
        (field[1:-1, 1] - 2 * field[1:-1, 0] + field[1:-1, 1]) / dy**2
    )

    # Right boundary
    lap[1:-1, -1] = (
        (field[2:, -1] + field[:-2, -1] - 2 * field[1:-1, -1]) / dx**2 +
        (field[1:-1, -2] - 2 * field[1:-1, -1] + field[1:-1, -2]) / dy**2
    )

    # Corners (if needed, but often not necessary in many physics problems)
    lap[0, 0] = lap[0, 1]
    lap[0, -1] = lap[0, -2]
    lap[-1, 0] = lap[-1, 1]
    lap[-1, -1] = lap[-1, -2]

    return lap


##########################
# Successive Overrelaxation (SOR) #
##########################
def sor(P, f, dx=dx, dy=dy, omega=omega, tol=tolerance, max_iter=max_iter):
    """Vectorized Successive Over-Relaxation (SOR)."""
    coef = 1 / (2 / dx**2 + 2 / dy**2)
    for it in range(max_iter):
        P_old = P.copy()
        P[1:-1, 1:-1] = (
            (1 - omega) * P[1:-1, 1:-1] +
            omega * coef * (
                (P[2:, 1:-1] + P[:-2, 1:-1]) / dx**2 +
                (P[1:-1, 2:] + P[1:-1, :-2]) / dy**2 - f[1:-1, 1:-1]
            )
        )
        if np.linalg.norm(P - P_old, ord=np.inf) < tol:
            print(f"SOR converged after {it+1} iterations.")
            break
    else:
        print("SOR did not converge within the maximum iterations.")
    return P

##########################
# Fractional-Step Method #
##########################
for it in range(max_iter):
    # Boundary conditions for u, v
    u[:, 0] = u[:, -1] = 0
    u[0, :] = u[-1, :] = 0
    v[0, :int(L_slot / Lx * nb_points)] = -U_slot
    v[-1, :int(L_slot / Lx * nb_points)] = U_slot
    v[0, int(L_slot / Lx * nb_points):int((L_slot + L_coflow) / Lx * nb_points)] = -U_coflow
    v[-1, int(L_slot / Lx * nb_points):int((L_slot + L_coflow) / Lx * nb_points)] = U_coflow

    # Step 1: Predictor velocities
    u_star = u - dt * (u * np.gradient(u, dx, axis=0) + v * np.gradient(u, dy, axis=1))
    v_star = v - dt * (u * np.gradient(v, dx, axis=0) + v * np.gradient(v, dy, axis=1))

    # Step 2: Diffusion step
    u_double_star = u_star + dt * nu * laplacian(u_star)
    v_double_star = v_star + dt * nu * laplacian(v_star)

    # Step 3: Pressure correction
    rhs = rho / dt * (np.gradient(u_double_star, dx, axis=0) + np.gradient(v_double_star, dy, axis=1))
    P = sor(P, rhs)

    # Step 4: Correct velocities
    u_new = u_double_star - dt / rho * np.gradient(P, dx, axis=0)
    v_new = v_double_star - dt / rho * np.gradient(P, dy, axis=1)

    # Update fields
    u, v = u_new, v_new

    # Convergence check (optional)
    if it % 100 == 0:
        print(f"Iteration {it}: max |u| = {np.max(np.abs(u)):.6f}, max |v| = {np.max(np.abs(v)):.6f}")

# Calculate maximum strain rate
strain_rate = np.abs(-np.gradient(u, dx, axis=0))
max_strain_rate = np.max(strain_rate)

print(f"Maximum strain rate (|∂v/∂y|) on left wall: {max_strain_rate:.6f} s^-1")
