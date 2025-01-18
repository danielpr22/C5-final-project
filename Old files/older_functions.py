import numpy as np
from numba import jit
import derivatives as der
from final_project_rk4 import sor, boundary_conditions, T 
from constants import (
    nb_points, dt, nb_timesteps, Lx, Ly, L_slot, L_coflow, U_slot, U_coflow, 
    T_slot, T_coflow, dx, dy, max_iter_sor, omega, tolerance_sor, tolerance_sys, 
    rho, nu, A, T_a, c_p, W_N2, W_O2, W_CH4, W_H2O, W_CO2, 
    nu_ch4, nu_o2, nu_n2, nu_h2o, nu_co2, h_n2, h_o2, h_ch4, h_h2o, h_co2
)



@jit(fastmath=True, nopython=True)
def update_Y_k(Y_k : np.array, u: np.array, v: np.array, source_term: np.array):
    """The advection terms are solved following central differences, and the diffusion terms
    following second-order central differences
    """
    Y_k_updated = np.zeros_like(Y_k)
    derivative_x = der.derivative_x_centered(Y_k)
    derivative_y = der.derivative_y_centered(Y_k)
    second_derivative_x = der.second_centered_x(Y_k)
    second_derivative_y = der.second_centered_y(Y_k)

    for i in (1, nb_points-1): 
        for j in (1, nb_points-1):
            Y_k_updated[i, j] = Y_k[i, j] - dt * (u[i, j] * derivative_x[i,j] + v[i, j] * derivative_y[i, j]) + dt * nu * (second_derivative_x[i, j] + second_derivative_y[i, j]) + dt * source_term[i, j]

    return Y_k_updated


@jit(fastmath=True, nopython=True)
def system_evolution_kernel(u, v, P, Y_n2, Y_o2, Y_ch4):
    
    # Step 1
    u_star = u - dt * (u * der.derivative_x_centered(u) + v * der.derivative_y_centered(u))
    v_star = v - dt * (u * der.derivative_x_centered(v) + v * der.derivative_y_centered(v))
    
    # Species transport 
    concentration_ch4 = rho / W_CH4 * Y_ch4
    concentration_o2 = rho / W_O2 * Y_o2
    Q = A * concentration_ch4 * concentration_o2**2 * np.exp(-T_a / T)
    source_term_n2 = nu_n2 / rho * W_N2 * Q
    source_term_o2 = nu_o2 / rho * W_O2 * Q
    source_term_ch4 = nu_ch4 / rho * W_CH4 * Q

    Y_n2_new = update_Y_k(Y_n2, u, v, source_term_n2)

    Y_o2_new = update_Y_k(Y_o2, u, v, source_term_o2)

    Y_ch4_new = update_Y_k(Y_ch4, u, v, source_term_ch4)


    # Step 2
    u_double_star = u_star + dt * (nu * (der.second_centered_x(u_star) + der.second_centered_y(u_star)))
    v_double_star = v_star + dt * (nu * (der.second_centered_x(v_star) + der.second_centered_y(v_star)))


    # Step 3 (P is updated)
    P = sor(P, f=rho / dt * (der.derivative_x_centered(u_double_star) + der.derivative_y_centered(v_double_star)))

    # Step 4
    u_new = u_double_star - dt / rho * der.derivative_x_centered(P)
    v_new = v_double_star - dt / rho * der.derivative_y_centered(P)

    u_new, v_new, Y_n2_new, Y_o2_new, Y_ch4_new = boundary_conditions(u_new, v_new, Y_n2_new, Y_o2_new, Y_ch4_new)

    P[:, 0] = 0
    P[:, 1] = P[:, 2] # dP/dx = 0 at the left wall
    P[0, 1:] = P[1, 1:] # dP/dy = 0 at the top wall
    P[-1, 1:] = P[-2, 1:] # dP/dy = 0 at the bottom wall
    P[:, -1] = 0 # P = 0 at the right free limit

    return u_new, v_new, P, Y_n2_new, Y_o2_new, Y_ch4_new


@jit(fastmath=True, nopython=True)
def compute_rhs_u_field(u, v, nu, dx, dy):
    """
    Compute the RHS of the advection-diffusion equation for u.
    """
    adv_x = u * der.derivative_x_centered(u, dx)
    adv_y = v * der.derivative_y_centered(u, dy)
    diff_x = nu * der.second_centered_x(u, dx)
    diff_y = nu * der.second_centered_y(u, dy)
    return -(adv_x + adv_y) + (diff_x + diff_y)


@jit(fastmath=True, nopython=True)
def compute_rhs_v_field(u, v, nu=nu, dx=dx, dy=dy):
    """
    Compute the RHS of the advection-diffusion equation for v.
    """
    adv_x = u * der.derivative_x_centered(v, dx)
    adv_y = v * der.derivative_y_centered(v, dy)
    diff_x = nu * der.second_centered_x(v, dx)
    diff_y = nu * der.second_centered_y(v, dy)
    return -(adv_x + adv_y) + (diff_x + diff_y)


@jit(fastmath=True, nopython=True)
def rk4_velocity(u, v, nu, dx, dy, dt):
    """
    Perform a single RK4 time step for the velocity fields u and v.
    """
    # RK4 for u
    k1_u = dt * compute_rhs_u_field(u, v, nu, dx, dy)
    k2_u = dt * compute_rhs_u_field(u + 0.5 * k1_u, v, nu, dx, dy)
    k3_u = dt * compute_rhs_u_field(u + 0.5 * k2_u, v, nu, dx, dy)
    k4_u = dt * compute_rhs_u_field(u + k3_u, v, nu, dx, dy)
    u_new = u + (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6

    # RK4 for v
    k1_v = dt * compute_rhs_v_field(u, v, nu, dx, dy)
    k2_v = dt * compute_rhs_v_field(u, v + 0.5 * k1_v, nu, dx, dy)
    k3_v = dt * compute_rhs_v_field(u, v + 0.5 * k2_v, nu, dx, dy)
    k4_v = dt * compute_rhs_v_field(u, v + k3_v, nu, dx, dy)
    v_new = v + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

    return u_new, v_new