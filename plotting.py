import matplotlib.pyplot as plt
import os
import numpy as np
from constants import (
    nb_points, dt, nb_timesteps, Lx, Ly, L_slot, L_coflow, U_slot, U_coflow, 
    T_slot, T_coflow, dx, dy, max_iter_sor, omega, tolerance_sor, tolerance_sys, 
    rho, nu, A, T_a, c_p, W_N2, W_O2, W_CH4, W_H2O, W_CO2, 
    nu_ch4, nu_o2, nu_n2, nu_h2o, nu_co2, h_n2, h_o2, h_ch4, h_h2o, h_co2
)

# Meshgrid for the plots

x,y = np.linspace(0, Lx, nb_points), np.linspace(Ly, 0, nb_points) # For the quiver methods 
X,Y = np.meshgrid(x,y,indexing='xy')

######################################################
# Plotting and animating the flow and the velocities #
######################################################


def plot_velocity_fields(u, v, Lx, Ly, show_figures, dpi, filename, output_folder=None, L_slot=L_slot, L_coflow=L_coflow):
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
        output_folder (str, optional): Path to save the plots
    """

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

    output_folder = os.path.join(output_folder, filename)
    plt.savefig(output_folder, dpi = dpi)

    if show_figures: 
        plt.show()
    else:
        plt.close()


def plot_vector_field(u: np.array, v: np.array, output_folder, dpi, filename='Flow field'):
    output_folder = os.path.join(output_folder, filename)

    plt.quiver(X*1e3,Y*1e3,u,v,scale=10)
    plt.title('Initial velocity field configuration')
    plt.xlabel('Lx (mm)')
    plt.ylabel('Ly (mm)')
    plt.savefig(output_folder, dpi = dpi)
    plt.close()



def animate_flow_evolution(u_history, v_history, Lx, Ly,
                           nb_points, L_slot, L_coflow, output_folder, 
                           dpi, interval=100, skip_frames=5):
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
        output_folder : Path where the data will be stored
        dpi : For quality
        interval (int): Interval between frames in milliseconds (default: 200ms for slower animation)
        skip_frames (int): Number of frames to skip between each animation frame (default: 2)
        output_folder (str, optional): Path to save the animation
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

    filename = 'Time animation of the velocity fields.gif'
    if output_folder:
        anim.save(os.path.join(output_folder, filename), dpi = dpi, writer="pillow")
    
    plt.show()


def plot_field(Y_k : np.array, output_folder, show_figures, dpi, filename):

    # Create coordinate meshgrid with reversed y-axis for matrix indexing
    x = np.linspace(0, Lx, nb_points)
    y = np.linspace(Ly, 0, nb_points)  # Reversed for matrix indexing
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(6, 5))

    plt.imshow(Y_k, cmap='viridis')  # Display the data as an image
    plt.colorbar(label='Value')  # Add a colorbar with a label
    plt.title(filename)  # Add a title
    plt.xlabel('Lx (mm)')  # Label for the x-axis
    plt.ylabel('Ly(mm)')  # Label for the y-axis
    output_folder = os.path.join(output_folder, filename)
    plt.savefig(output_folder, dpi = dpi)

    if show_figures: 
        plt.show()
    else:
        plt.close()