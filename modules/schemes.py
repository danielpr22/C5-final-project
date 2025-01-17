
import numpy as np
from .constants import *
from .derivatives import *

omega = 2/(1+np.sin(np.pi/nb_points))

@jit(fastmath=True, nopython=True)
def sor(P, f, tol_sor=tol_sor, max_iter=max_iter, omega=omega):

    coef = 2 * (1 / dx**2 + 1 / dy**2)

    for _ in range(max_iter):
        P_old = P.copy()
        laplacian_P = np.zeros_like(P)

        for i in range(1, nb_points - 1):
            for j in range(1, nb_points - 1):
                laplacian_P[i, j] = (P_old[i+1, j] + P[i-1, j]) / dx**2 + (P_old[i, j+1] + P[i, j-1]) / dy**2
                
                # Update P using the SOR formula
                P[i, j] = (1 - omega) * P_old[i, j] + (omega / coef) * (laplacian_P[i, j] - f[i, j])
        
        # for i in range(1, nb_points - 1):
        #     for j in range(1, nb_points - 1):
        #         laplacian_P[i, j] = (P[i+1, j] + P[i-1, j]) / dx**2 + (P[i, j+1] + P[i, j-1]) / dy**2
                
        #         # Update P using the SOR formula
        #         P[i, j] = (1 - omega) * P[i, j] + (omega / coef) * (laplacian_P[i, j] - f[i, j])
        P[:, 0] = 0
        P[:, 1] = P[:, 2] # dP/dx = 0 at the left wall
        P[0, 1:] = P[1, 1:] # dP/dy = 0 at the top wall
        P[-1, 1:] = P[-2, 1:] # dP/dy = 0 at the bottom wall
        P[:, -1] = 0 # P = 0 at the right free limit P = 0 at the right free limit
        # Compute the residual to check for convergence
        # residual = np.linalg.norm(P - P_old, ord=2)
        residual= P-P_old
        # if np.linalg.norm(residual, ord=np.inf) > 10e-10:  # Avoid divide-by-zero by checking the norm
        #     residual /= np.linalg.norm(P_old, ord=2)

        if np.linalg.norm(residual, ord=np.inf) < tol_sor:
            return P
    print("SOR did not converge")
    return P

@jit(fastmath=True, nopython=True)
def lax_wendroff(u:np.array, v:np.array, P:np.array, Y_n2:np.array):
    u, v, Y_n2 = boundary_conditions(u, v, Y_n2)

    # Step 1: Calculate intermediate values at n+1/2
    # Using centered differences for spatial derivatives
    u_half = u - 0.5*dt * (u * derivative_x_centered(u) + v * derivative_y_centered(u))
    v_half = v - 0.5*dt * (u * derivative_x_centered(v) + v * derivative_y_centered(v))
    Y_n2_half = Y_n2 - 0.5*dt * (u * derivative_x_centered(Y_n2) + v * derivative_y_centered(Y_n2))

    # Apply boundary conditions to half-step values
    u_half, v_half, Y_n2_half = boundary_conditions(u_half, v_half, Y_n2_half)

    # Add viscous terms for half step
    u_half = u_half + 0.5*dt * nu * (second_centered_x(u) + second_centered_y(u))
    v_half = v_half + 0.5*dt * nu * (second_centered_x(v) + second_centered_y(v))
    Y_n2_half = Y_n2_half + 0.5*dt * nu * (second_centered_x(Y_n2) + second_centered_y(Y_n2))

    # Step 2: Use these half-step values to compute the full step
    u_star = u - dt * (u_half * derivative_x_centered(u_half) + v_half * derivative_y_centered(u_half))
    v_star = v - dt * (u_half * derivative_x_centered(v_half) + v_half * derivative_y_centered(v_half))
    Y_n2_star = Y_n2 - dt * (u_half * derivative_x_centered(Y_n2_half) + v_half * derivative_y_centered(Y_n2_half))

    # Add viscous terms for full step
    u_double_star = u_star + dt * nu * (second_centered_x(u_half) + second_centered_y(u_half))
    v_double_star = v_star + dt * nu * (second_centered_x(v_half) + second_centered_y(v_half))
    Y_n2_double_star = Y_n2_star + dt * nu * (second_centered_x(Y_n2_half) + second_centered_y(Y_n2_half))

    # Apply boundary conditions
    u_double_star, v_double_star, Y_n2_double_star = boundary_conditions(u_double_star, v_double_star, Y_n2_double_star)

    # Pressure correction step (same as before)
    P = sor(P, f=rho/dt * (derivative_x_centered(u_double_star) + derivative_y_centered(v_double_star)))

    # Final velocity correction
    u_new = u_double_star - dt/rho * derivative_x_centered(P)
    v_new = v_double_star - dt/rho * derivative_y_centered(P)

    # Final boundary conditions
    u_new, v_new, Y_n2_double_star = boundary_conditions(u_new, v_new, Y_n2_double_star)

    return u_new, v_new, P, Y_n2_double_star

@jit(fastmath=True, nopython=True)
def euler(u:np.array, v:np.array, P:np.array, Y_n2:np.array):
    
    u, v, Y_n2 = boundary_conditions(u, v, Y_n2)

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
    # Step 3 (P is updated)
    P = sor(P, f=rho / dt * (derivative_x_centered(u_double_star) + derivative_y_centered(v_double_star)))

    
   
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
