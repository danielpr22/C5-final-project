import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 0.002, 0.002  # Domain size (meters)
Nx, Ny = 50, 50        # Grid points
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # Grid spacing
nu = 15e-6             # Kinematic viscosity (m^2/s)
U_slot = 1.0           # Slot velocity (m/s)
tolerance = 1e-6       # Convergence criterion

# Initialize fields
u = np.zeros((Nx, Ny))  # x-velocity
v = np.zeros((Nx, Ny))  # y-velocity
p = np.zeros((Nx, Ny))  # Pressure
u_new = np.copy(u)
v_new = np.copy(v)

# Boundary conditions
u[:, 0] = 0            # Bottom wall (no-slip)
u[:, -1] = 0           # Top wall (no-slip)
u[0, :] = U_slot       # Left inlet velocity (slipping wall)
u[-1, :] = 0           # Right wall (no-slip)

# Solver parameters
dt = 0.001  # Time step (s)
max_iter = 1000

# Helper functions
def laplacian(field, dx, dy):
    """Compute 2D Laplacian using central differences."""
    lap = (
        (np.roll(field, -1, axis=0) - 2 * field + np.roll(field, 1, axis=0)) / dx**2 +
        (np.roll(field, -1, axis=1) - 2 * field + np.roll(field, 1, axis=1)) / dy**2
    )
    return lap

# Time-marching loop for steady-state solution
for it in range(max_iter):
    # Momentum equations (explicit Euler)
    u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (nu * laplacian(u, dx, dy))
    v_new[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (nu * laplacian(v, dx, dy))

    # Update boundary conditions
    u_new[0, :] = u_new[1, :]  # Free slip on left wall
    u_new[:, 0] = 0           # Bottom wall (no-slip)
    u_new[:, -1] = 0          # Top wall (no-slip)
    u_new[-1, :] = 0          # Right wall (no-slip)

    # Convergence check
    if np.max(np.abs(u_new - u)) < tolerance:
        print(f"Converged after {it} iterations.")
        break

    u, v = u_new.copy(), v_new.copy()

# Compute strain rate on the left wall
strain_rate = np.abs(np.gradient(v[:, 0], dy))

# Find maximum strain rate
max_strain_rate = np.max(strain_rate)
print(f"Maximum strain rate (|∂v/∂y|) on left wall: {max_strain_rate:.6f} s^-1")

# Plot strain rate along the left wall
plt.plot(np.linspace(0, Ly, Ny), strain_rate, label="Strain rate |∂v/∂y|")
plt.xlabel("y (m)")
plt.ylabel("Strain rate (s^-1)")
plt.title("Strain Rate Along the Left Wall")
plt.legend()
plt.grid()
plt.show()
