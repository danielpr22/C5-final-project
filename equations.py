# Fluid flow equations

import numpy as np
from constants import constants as c


def fractional_step(u: np.array, v: np.array, P: np.array, dt = c.dt) -> np.array:     

    # Use slices for i and j instead of a for loop for vectorization
    for i in range(0, 1):
        for j  in range(0, c.nb_points):
            us= u[i,j] - dt * (u[i, j] * (derivative_x(u, u, i, j) + v[i, j] * derivative_y(u, v, i, j)))
    
    uss = us + dt * c.eta()


def derivative_x(u:np.array, v:np.array, i:int, j:int, dx = c.dx) -> np.array: 
    return (u[i, j] - u[i - 1, j]) / dx if u[i, j] > 0 else (u[i + 1, j] - u[i, j]) / dx

def derivative_y(u:np.array, v:np.array, i:int, j:int, dy = c.dy) -> np.array: 
    return (u[i, j] - u[i, j - 1]) / dy if v[i, j] > 0 else (u[i, j + 1] - u[i, j]) / dy

def second_centered_x(u:np.array, i:int, j:int, dx = c.dx) -> np.array: 
    return (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx**2

def second_centered_y(u:np.array, i:int, j:int, dy = c.dy) -> np.array: 
    return (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy**2