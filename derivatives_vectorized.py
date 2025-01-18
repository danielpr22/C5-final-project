from numba import jit
import numpy as np
from constants import (
    nb_points, dt, nb_timesteps, Lx, Ly, L_slot, L_coflow, U_slot, U_coflow, 
    T_slot, T_coflow, dx, dy, max_iter_sor, omega, tolerance_sor, tolerance_sys, 
    rho, nu, A, T_a, c_p, W_N2, W_O2, W_CH4, W_H2O, W_CO2, 
    nu_ch4, nu_o2, nu_n2, nu_h2o, nu_co2, h_n2, h_o2, h_ch4, h_h2o, h_co2
)


#################################################
# Derivatives and second derivatives definition #
#################################################

@jit(fastmath=True, nopython=True)
def derivative_x_centered(u: np.array, dx=dx) -> np.array:
    """
    Computes the centered x-derivative of a 2D array u using vectorized operations.
    """
    du_dx = np.zeros_like(u)
    du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)  # Central difference
    return du_dx


@jit(fastmath=True, nopython=True)
def derivative_y_centered(u: np.array, dy=dy) -> np.array:
    """
    Computes the centered y-derivative of a 2D array u using vectorized operations.
    """
    du_dy = np.zeros_like(u)
    du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)  # Central difference
    return du_dy


@jit(fastmath=True, nopython=True)
def derivative_x_upwind(vel: np.array, a: np.array, dx=dx) -> np.array:
    """
    Computes the x-derivative of a 2D array vel using the upwind scheme and vectorized operations.
    """
    dvel_dx = np.zeros_like(vel)
    positive_flow = a > 0
    dvel_dx[:, 1:-1] = np.where(
        positive_flow[:, 1:-1],
        (vel[:, 1:-1] - vel[:, :-2]) / dx,  # Upwind difference for positive flow
        (vel[:, 2:] - vel[:, 1:-1]) / dx   # Upwind difference for negative flow
    )
    return dvel_dx


@jit(fastmath=True, nopython=True)
def derivative_y_upwind(vel: np.array, a: np.array, dy=dy) -> np.array:
    """
    Computes the y-derivative of a 2D array vel using the upwind scheme and vectorized operations.
    """
    dvel_dy = np.zeros_like(vel)
    positive_flow = a > 0
    dvel_dy[1:-1, :] = np.where(
        positive_flow[1:-1, :],
        (vel[1:-1, :] - vel[:-2, :]) / dy,  # Upwind difference for positive flow
        (vel[2:, :] - vel[1:-1, :]) / dy    # Upwind difference for negative flow
    )
    return dvel_dy


@jit(fastmath=True, nopython=True)
def second_centered_x(u: np.array, dx=dx) -> np.array:
    """
    Computes the second centered x-derivative of a 2D array u using vectorized operations.
    """
    d2u_dx2 = np.zeros_like(u)
    d2u_dx2[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / dx**2
    return d2u_dx2


@jit(fastmath=True, nopython=True)
def second_centered_y(u: np.array, dy=dy) -> np.array:
    """
    Computes the second centered y-derivative of a 2D array u using vectorized operations.
    """
    d2u_dy2 = np.zeros_like(u)
    d2u_dy2[1:-1, :] = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / dy**2
    return d2u_dy2