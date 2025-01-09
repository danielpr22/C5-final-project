import sys
import os
import string
import numpy as np
import matplotlib.pyplot as plt

# We had a problem with the path of the other module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#############
# Constants #
#############

nb_points = 100

R = 8314 # Ideal gas constant in J/(kg * K)

dt = 1e-4

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

# Kinematic viscosity
eta = 15e-6

dx = Lx / (nb_points - 1)

dy = Ly / (nb_points - 1)

nu = 15e-6

tolerance = 1e-6

max_iter = 1000 # Maximum number of iterations

P = np.zeros((nb_points, nb_points))  # Pressure


############################################
# Initial conditions & boundary conditions #
############################################
v = np.zeros((nb_points, nb_points))
v[0, 0:int(L_slot/Lx * nb_points)] = -U_slot # Speed at the top of the "flow" region (Methane)
v[-1, 0:int(L_slot/Lx * nb_points)] = U_slot # Speed at the bottom of the "flow" region (Air)

v[0, int(L_slot/Lx * nb_points):int((L_slot + L_coflow)/Lx * nb_points)] = - U_coflow # Speed at the top of the "coflow" region (Nitrogen)
v[-1, int(L_slot/Lx * nb_points):int((L_slot + L_coflow)/Lx * nb_points)] = U_coflow  # Speed at the top of the "coflow" region (Nitrogen)

P[0, 0:int(L_slot/Lx * nb_points)] = rho * R * T_slot
P[-1, 0:int(L_slot/Lx * nb_points)] = rho * R * T_slot

P[0, int(L_slot/Lx * nb_points):int((L_slot + L_coflow)/Lx * nb_points)] = rho * R * T_coflow
P[-1, int(L_slot/Lx * nb_points):int((L_slot + L_coflow)/Lx * nb_points)] = rho * R * T_coflow


u = np.zeros((nb_points, nb_points))
    
u_star = np.copy(u)
v_star = np.copy(v)


#################################################
# Derivatives and second derivatives definition #
#################################################
def derivative_x(u: np.array, var_name : string, dx=dx) -> np.array:
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
    u_left = np.roll(u, shift=1, axis=0)   # u[i-1, j]
    u_right = np.roll(u, shift=-1, axis=0) # u[i+1, j]
    
    # Apply upwind scheme based on the sign of u

    derivative = np.where(u > 0, (u - u_left) / dx, (u_right - u) / dx)

    if var_name == "u":
        derivative[:,-1] = 0 # Right free limit condition
 
    if var_name == "v": 
        derivative[:,0] = 0 # Left slippery wall
        derivative[:,-1] = 0 # Right free limit condition
    
    if var_name == "P":
        derivative[:,0] = 0 # Left slippery wall

    return derivative

def derivative_y(u: np.array, var_name : string, dy=dy) -> np.array:
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
    u_left = np.roll(u, shift=1, axis=1)   # u[i, j - 1]
    u_right = np.roll(u, shift=-1, axis=1) # u[i, j + 1]
    
    # Apply upwind scheme based on the sign of u
    derivative = np.where(u > 0, (u - u_left) / dy, (u_right - u) / dy)

    if var_name == "P":
        derivative[0,:] = 0
        derivative[-1,:] = 0

    return derivative


def second_centered_x(u:np.array, dx = dx) -> np.array: 

    u_left = np.roll(u, shift=1, axis=0)   # u[i-1, j]
    u_right = np.roll(u, shift=-1, axis=0) # u[i+1, j]
    
    return (u_right - 2 * u + u_left) / dx**2

def second_centered_y(u:np.array, dy = dy) -> np.array: 

    u_left = np.roll(u, shift=1, axis=1)   # u[i, j - 1]
    u_right = np.roll(u, shift=-1, axis=1) # u[i, j + 1]

    return (u_right - 2 * u + u_left) / dy**2


###########################
# Navier-Stokes equations #
###########################

for it in range(max_iter):
    # Navier-Stokes equations
    u_star[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (+ u[1:-1, 1:-1] * derivative_y(v, "v")[1:-1, 1:-1] - v[1:-1, 1:-1] * derivative_y(u, "u")[1:-1, 1:-1] - 1/rho * derivative_x(P, "P")[1:-1, 1:-1] + nu * (second_centered_x(u)[1:-1, 1:-1] + second_centered_y(u)[1:-1, 1:-1]))
    v_star[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (- u[1:-1, 1:-1] * derivative_x(v, "v")[1:-1, 1:-1] - v[1:-1, 1:-1] * derivative_y(v, "v")[1:-1, 1:-1] - 1/rho * derivative_y(P, "P")[1:-1, 1:-1] + nu * (second_centered_x(v)[1:-1, 1:-1] + second_centered_y(v)[1:-1, 1:-1])) 


    # Update boundary conditions

    #u_star[0, int((L_slot + L_coflow)/Lx * nb_points):nb_points] = 0    # Top wall after the coflow(no-slip)
    #u_star[-1, int((L_slot + L_coflow)/Lx * nb_points):nb_points] = 0   # Bottom wall after the coflow (no-slip)

    # Convergence check
    if np.max(np.abs(v_star - v)) < tolerance:
        print(f"Converged after {it} iterations.")
        break

    u, v = np.copy(u_star), np.copy(v_star) # Updating of the fields at each iteration

# Once the u field is correctly advected, we calculate "a"
# Conservation of mass
strain_rate = np.abs(-derivative_x(u, "u")) # Since dv/dy = -du/dx

max_strain_rate = np.max(strain_rate)

print(f"Maximum strain rate (|∂v/∂y|) on left wall: {max_strain_rate:.6f} s^-1")


# Plot strain rate along the left wall
plt.plot(np.linspace(0, Ly, nb_points), strain_rate, label="Strain rate |∂v/∂y|")
plt.xlabel("y (m)")
plt.ylabel("Strain rate (s^-1)")
plt.title("Strain Rate Along the Left Wall")
plt.legend()
plt.grid()
plt.show()