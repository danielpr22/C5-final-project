import numpy as np
import matplotlib.pyplot as plt


def v_map(u: np.array, v: np.array, X: np.array, Y: np.array, color: str):

    plt.streamplot(X, Y, -u, v, color=color)
    plt.title("velocity field lines", fontsize=22)
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)


def streamplot(ax, u, v, Lx, Ly, nb_points):
    x = np.linspace(0, Lx, nb_points)
    y = np.linspace(Ly, 0, nb_points)
    X, Y = np.meshgrid(x, y, indexing="xy")

    ax.quiver(X, Y, -u, v, color="black")
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Streamplot of Vector Velocity Field")


def P(ax, P, Lx, Ly, vmin, vmax):
    # Use pcolormesh for consistency with the streamplot
    P = P
    pressure_plot = ax.imshow(
        -P,
        extent=(0, Lx, 0, Ly),
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="gaussian",
    )
    return pressure_plot  # Return handle for colorbar


def animate_flow_evolution(
    u_history,
    v_history,
    Lx,
    Ly,
    nb_points,
    L_slot,
    L_coflow,
    interval=1500,
    skip_frames=2,
    save_path="C:\\Users\\danie\\Desktop\\animation.gif",
):
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
        interval (int): Interval between frames in milliseconds (default: 200ms for slower animation)
        skip_frames (int): Number of frames to skip between each animation frame (default: 2)
        save_path (str, optional): Path to save the animation
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

    if save_path:
        anim.save(save_path, writer="pillow")

    plt.show()


def velocity_fields(u, v, Lx, Ly, nb_points, L_slot, L_coflow, save_path=None):
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
        save_path (str, optional): Path to save the plots
    """
    # Create coordinate meshgrid
    x = np.linspace(0, Lx, nb_points)
    y = np.linspace(Ly, 0, nb_points)  # Reversed for matrix indexing
    X, Y = np.meshgrid(x, y, indexing="xy")

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

    if save_path:
        plt.savefig(save_path)

    plt.show()
