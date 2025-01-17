
import numpy as np
from numba import jit

def get(
    dt=1e-7, final_time=1, tol_sor=1e-8, tol_sys=1e-7, max_iter=10000, nb_points=42
):
    R = 8.314  # Ideal gas constant in J/(mol* K)
    dt = dt
    final_time = final_time
    nb_timesteps = int(final_time / dt)
    Lx = 2e-3
    Ly = 2e-3
    L_slot = 0.5e-3
    L_coflow = 0.5e-3
    U_slot = 1.0
    T_slot = 300
    U_coflow = 0.2
    T_coflow = 300
    rho = 1.1614  # Fluid density
    nu = 15e-6
    tol_sor = tol_sor  # Tolerance for the convergence of the SOR algorithm
    tol_sys = tol_sys  # Tolerance for the convergence of the whole system
    max_iter = max_iter
    nb_points = nb_points

    return (
        R,
        dt,
        final_time,
        nb_timesteps,
        Lx,
        Ly,
        L_slot,
        L_coflow,
        U_slot,
        T_slot,
        U_coflow,
        T_coflow,
        rho,
        nu,
        tol_sor,
        tol_sys,
        max_iter,
        nb_points,
    )


(
    R,
    dt,
    final_time,
    nb_timesteps,
    Lx,
    Ly,
    L_slot,
    L_coflow,
    U_slot,
    T_slot,
    U_coflow,
    T_coflow,
    rho,
    nu,
    tol_sor,
    tol_sys,
    max_iter,
    nb_points,
) = get()


@jit(fastmath=True, nopython=True)
def boundary_conditions(u:np.array, v:np.array, Y_n2:np.array):
    # Boundary conditions for the velocity field
    # Boundary conditions for the velocity field
    u[:, 0] = 0
    u[:, 1] = 0  # Left slipping wall
    u[0, :] = 0  # Top non-slipping wall
    u[-1, :] = 0  # Bottom non-slipping wall
    u[:, -1] = u[:, -2]  # du/dx = 0 at the right free limit

    v[:, 0] = 0  # Left wall
    v[0, 1 : int(L_slot / Lx * nb_points) + 1] = (
        -U_slot
    )  # Speed at the top of the "flow" region (Methane)
    v[-1, 1 : int(L_slot / Lx * nb_points) + 1] = (
        U_slot  # Speed at the bottom of the "flow" region (Air)
    )
    v[
        0, int(L_slot / Lx * nb_points) + 1 : int((L_slot + L_coflow) / Lx * nb_points)
    ] = -U_coflow  # Speed at the top of the "coflow" region (Nitrogen)
    v[
        -1, int(L_slot / Lx * nb_points) + 1 : int((L_slot + L_coflow) / Lx * nb_points)
    ] = U_coflow  # Speed at the bottom of the "coflow" region (Nitrogen)
    v[0, int((L_slot + L_coflow) / Lx * nb_points) : nb_points] = (
        0  # For the top non-slipping wall
    )
    v[-1, int((L_slot + L_coflow) / Lx * nb_points) : nb_points] = (
        0  # For the bottom non-slipping wall
    )
    v[:, 1] = v[:, 2]  # dv/dx = 0 at the left wall
    v[:, -1] = v[:, -2]  # dv/dx = 0 at the right free limit

    Y_n2[-1, 1 : int(L_slot / Lx * nb_points) + 1] = (
        0.767  # Initial condition for the nitrogen in air (bottom slot)
    )
    Y_n2[
        0, int(L_slot / Lx * nb_points) : int((L_slot + L_coflow) / Lx * nb_points)
    ] = 1
    Y_n2[
        -1, int(L_slot / Lx * nb_points) : int((L_slot + L_coflow) / Lx * nb_points)
    ] = 1

    return u, v, Y_n2
