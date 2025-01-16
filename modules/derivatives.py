from numba import jit
import numpy as np

Lx = 2e-3
Ly = 2e-3
nb_points = 42
dx = Lx / (nb_points - 1)
dy = Ly / (nb_points - 1)


@jit(fastmath=True, nopython=True)
def derivative_x_upwind(vel: np.array, a:np.array, dx=dx) -> np.array:
    dvel_dx = np.zeros_like(vel)

    # For interior points, choose upwind direction based on the sign of u
    for i in range(1, vel.shape[1] - 1): # Skip the boundaries
        dvel_dx[:, i] = np.where(a[:, i] > 0, (vel[:, i] - vel[:, i - 1]) / dx, (vel[:, i + 1] - vel[:, i]) / dx)

    return dvel_dx


@jit(fastmath=True, nopython=True)
def derivative_y_upwind(vel: np.array, a:np.array, dy=dy) -> np.array:
    dvel_dy = np.zeros_like(vel)

    # For interior points, choose upwind direction based on the sign of vel
    for i in range(1, vel.shape[0] - 1): 
        # Differenciate between positive and negative flow
        dvel_dy[i, :] = np.where(a[i, :] > 0, (vel[i, :] - vel[i - 1, :]) / dy, (vel[i + 1, :] - vel[i, :]) / dy)

    return dvel_dy


@jit(fastmath=True, nopython=True)
def derivative_x_centered(u: np.array, dx=dx) -> np.array:
    du_dx = np.zeros_like(u)

    for i in range(1, nb_points - 1):
        for j in range(1, nb_points - 1):
            du_dx[i, j] = (u[i, j+1] - u[i, j-1]) / (2 * dx)  # Central difference

    return du_dx


@jit(fastmath=True, nopython=True)
def derivative_y_centered(u: np.array, dy=dy) -> np.array:
    du_dy = np.zeros_like(u)

    for i in range(1, nb_points - 1):
        for j in range(1, nb_points - 1):
            du_dy[i, j] = (u[i+1, j] - u[i-1, j]) / (2 * dy)  # Central difference

    return du_dy


@jit(fastmath=True, nopython=True)
def second_centered_x(u: np.array, dx=dx) -> np.array:
    d2u_dx2 = np.zeros_like(u)

    for i in range(1, nb_points - 1):
        for j in range(1, nb_points - 1):
            u_left = u[i, j-1]
            u_right = u[i, j+1]
            d2u_dx2[i, j] = (u_right - 2 * u [i, j] + u_left) / dx**2

    return d2u_dx2


@jit(fastmath=True, nopython=True)
def second_centered_y(u: np.array, dy=dy) -> np.array:
    d2u_dy2 = np.zeros_like(u)

    for i in range(1, nb_points - 1):
        for j in range(1, nb_points - 1):
            u_left = u[i-1, j]
            u_right = u[i+1, j]
            d2u_dy2[i, j] = (u_right - 2 * u [i, j] + u_left) / dy**2

    return d2u_dy2