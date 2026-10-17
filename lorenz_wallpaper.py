
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

# ============================================================
# Configuration
# ============================================================

WIDTH: int = 1920
HEIGHT: int = 1080

# Render at 2x resolution, then downsample for cleaner antialiasing
SCALE: int = 2
RENDER_W: int = WIDTH * SCALE
RENDER_H: int = HEIGHT * SCALE

BG: str = "#0b1020"
CYAN: str = "#00c7d8"
GOLD: str = "#e5ad00"

# Set the counter-clockwise rotation angle here
ROTATION_ANGLE: float = 20.0

# Lorenz parameters
SIGMA: float = 10.0
RHO: float = 28.0
BETA: float = 8.0 / 3.0

# Image version
VERSION: str = "2.2"


# ============================================================
# Helper Functions
# ============================================================

def rotate_2d(
    x: float | np.ndarray, 
    z: float | np.ndarray, 
    angle_degrees: float
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Rotates 2D coordinates counter-clockwise.

    Args:
        x (Union[float, np.ndarray]): The x-coordinate(s) to rotate.
        z (Union[float, np.ndarray]): The z-coordinate(s) to rotate.
        angle_degrees (float): The angle of rotation in degrees.

    Returns:
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]: 
            A tuple containing the rotated x and z coordinates.
    """
    angle_rad: float = np.radians(angle_degrees)
    c: float = np.cos(angle_rad)
    s: float = np.sin(angle_rad)
    return x * c - z * s, x * s + z * c


def lorenz(x: float, y: float, z: float) -> np.ndarray:
    """Computes the instantaneous derivatives for the Lorenz system.

    Args:
        x (float): The x-coordinate of the system state.
        y (float): The y-coordinate of the system state.
        z (float): The z-coordinate of the system state.

    Returns:
        np.ndarray: An array containing the derivatives [dx, dy, dz].
    """
    dx: float = SIGMA * (y - x)
    dy: float = x * (RHO - z) - y
    dz: float = x * y - BETA * z
    return np.array([dx, dy, dz], dtype=np.float64)


def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    """Performs one fourth-order Runge-Kutta integration step.

    Args:
        state (np.ndarray): The current state of the system [x, y, z].
        dt (float): The time step for integration.

    Returns:
        np.ndarray: The next state of the system after time step dt.
    """
    k1: np.ndarray = lorenz(*state)
    k2: np.ndarray = lorenz(*(state + 0.5 * dt * k1))
    k3: np.ndarray = lorenz(*(state + 0.5 * dt * k2))
    k4: np.ndarray = lorenz(*(state + dt * k3))

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def generate_trajectory(
    initial_state: tuple[float, float, float],
    dt: float = 0.005,
    burn_in: int = 3000,
    steps: int = 3000,
) -> np.ndarray:
    """Generates a Lorenz trajectory.
    
    Removes the initial transient (burn-in phase) so the trajectory
    settles onto the attractor before recording points.

    Args:
        initial_state (Tuple[float, float, float]): The starting (x, y, z) coordinates.
        dt (float, optional): The integration time step. Defaults to 0.005.
        burn_in (int, optional): Number of initial steps to discard. Defaults to 3000.
        steps (int, optional): Number of steps to record in the output. Defaults to 3000.

    Returns:
        np.ndarray: An array of shape (steps, 3) containing the trajectory coordinates.
    """
    state: np.ndarray = np.array(initial_state, dtype=np.float64)

    # Remove transient
    for _ in range(burn_in):
        state = rk4_step(state, dt)

    trajectory: np.ndarray = np.empty((steps, 3), dtype=np.float64)

    for i in range(steps):
        state = rk4_step(state, dt)
        trajectory[i] = state

    return trajectory


# ============================================================
# Generate many very similar trajectories
# ============================================================

seeds: list[tuple[float, float, float]] = [
    (0.1, 1.0, 1.05),
    (0.2, 1.0, 1.10),
    (0.3, 1.0, 1.15),
    (0.4, 1.0, 1.20),
    (0.5, 1.0, 1.25),
    (0.6, 1.0, 1.30),
    (0.7, 1.0, 1.35),
    (0.8, 1.0, 1.40),
    (0.9, 1.0, 1.45),
    (1.0, 1.0, 1.50),
    (1.1, 1.0, 1.55),
    (1.2, 1.0, 1.60),
    (1.3, 1.0, 1.65),
    (1.4, 1.0, 1.70),
]

trajectories: list[np.ndarray] = [generate_trajectory(seed) for seed in seeds]

# Apply scaling and rotation to trajectories
rotated_trajectories: list[tuple[np.ndarray, np.ndarray]] = []
for trajectory in trajectories:
    x: np.ndarray = trajectory[:, 0] * 1.00
    z: np.ndarray = trajectory[:, 2] * 0.98
    x_rot, z_rot = rotate_2d(x, z, ROTATION_ANGLE)
    rotated_trajectories.append((x_rot, z_rot))

# ============================================================
# Figure
# ============================================================

plt.rcParams["figure.facecolor"] = BG
plt.rcParams["axes.facecolor"] = BG

fig: plt.Figure = plt.figure(
    figsize=(RENDER_W / 100, RENDER_H / 100),
    dpi=100,
    facecolor=BG,
)

ax: plt.Axes = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(BG)

# ------------------------------------------------------------
# Set limits based on rotated coordinates
# ------------------------------------------------------------

all_x: np.ndarray = np.concatenate([t[0] for t in rotated_trajectories])
all_z: np.ndarray = np.concatenate([t[1] for t in rotated_trajectories])

x_min: float
x_max: float
z_min: float
z_max: float

x_min, x_max = np.percentile(all_x, [0.5, 99.5])
z_min, z_max = np.percentile(all_z, [0.5, 99.5])

# Slight expansion
x_pad: float = (x_max - x_min) * 0.06
z_pad: float = (z_max - z_min) * 0.08

x_min -= x_pad
x_max += x_pad
z_min -= z_pad
z_max += z_pad

ax.set_xlim(x_min, x_max)
ax.set_ylim(z_min, z_max)

# ------------------------------------------------------------
# Rotate and plot the central vertical axis & ticks
# ------------------------------------------------------------

unrot_x: np.ndarray = np.concatenate([t[:, 0] * 1.00 for t in trajectories])
unrot_z: np.ndarray = np.concatenate([t[:, 2] * 0.98 for t in trajectories])
ux_min: float
ux_max: float
uz_min: float
uz_max: float

ux_min, ux_max = np.percentile(unrot_x, [0.5, 99.5])
uz_min, uz_max = np.percentile(unrot_z, [0.5, 99.5])

# Pad original axis bounds
uz_min -= (uz_max - uz_min) * 0.08
uz_max += (uz_max - uz_min) * 0.08
ux_pad: float = (ux_max - ux_min) * 0.06

center_x: float = 0.0

# Rotate main axis line
rot_axis_x: np.ndarray
rot_axis_z: np.ndarray

rot_axis_x, rot_axis_z = rotate_2d(
    np.array([center_x, center_x]), 
    np.array([uz_min, uz_max]), 
    ROTATION_ANGLE
)

ax.plot(
    rot_axis_x,
    rot_axis_z,
    color="#192238",
    linewidth=1.0,
    alpha=0.9,
    zorder=0,
)

# Rotate tick marks
tick_values: np.ndarray = np.linspace(uz_min, uz_max, 7)
tick_width: float = (ux_max - ux_min + 2 * ux_pad) * 0.015

for y in tick_values:
    t_x: np.ndarray = np.array([center_x - tick_width, center_x + tick_width])
    t_y: np.ndarray = np.array([y, y])
    rt_x, rt_z = rotate_2d(t_x, t_y, ROTATION_ANGLE)
    
    ax.plot(
        rt_x,
        rt_z,
        color="#192238",
        linewidth=1.0,
        alpha=0.8,
        zorder=0,
    )

# ============================================================
# Draw rotated trajectories
# ============================================================

for i, (x_rot, z_rot) in enumerate(rotated_trajectories):
    # Alternate cyan and gold
    color: str = CYAN if i % 2 == 0 else GOLD

    ax.plot(
        x_rot,
        z_rot,
        color=color,
        linewidth=1.25 * SCALE / 2,
        alpha=0.90,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
    )

# ============================================================
# Add two highlighted points (Rotated)
# ============================================================

left_traj: np.ndarray = trajectories[4]
right_traj: np.ndarray = trajectories[7]

left_idx: int = int(len(left_traj) * 0.43)
right_idx: int = int(len(right_traj) * 0.72)

left_point: np.ndarray = left_traj[left_idx]
right_point: np.ndarray = right_traj[right_idx]

# Apply scaling and rotate the glowing points
lx_rot: float
lz_rot: float
rx_rot: float
rz_rot: float

lx_rot, lz_rot = rotate_2d(left_point[0] * 1.00, left_point[2] * 0.98, ROTATION_ANGLE)
rx_rot, rz_rot = rotate_2d(right_point[0] * 1.00, right_point[2] * 0.98, ROTATION_ANGLE)


def glow_point(
    x: float, 
    y: float, 
    color: str, 
    radius: int = 9
) -> None:
    """Draws a multi-layered glowing scatter point on the current axes.

    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        color (str): The hex color string for the glow.
        radius (int, optional): The base radius of the point. Defaults to 9.
    """
    # Outer glow
    ax.scatter(
        [x], [y], s=(radius * 4) ** 2,
        color=color, alpha=0.07, linewidths=0, zorder=5,
    )
    # Inner glow
    ax.scatter(
        [x], [y], s=(radius * 2.5) ** 2,
        color=color, alpha=0.12, linewidths=0, zorder=6,
    )
    # Core
    ax.scatter(
        [x], [y], s=radius**2,
        color=color, edgecolors="none", zorder=7,
    )


glow_point(lx_rot, lz_rot, CYAN, radius=7)
glow_point(rx_rot, rz_rot, "#ffc400", radius=7)


# ============================================================
# Remove everything except the artwork
# ============================================================

ax.set_axis_off()

# Save high-resolution intermediate
tmp_file: str = f"lorenz_wallpaper_{VERSION}_2x.png"

fig.savefig(
    tmp_file,
    dpi=100,
    facecolor=BG,
    edgecolor="none",
    bbox_inches=None,
    pad_inches=0,
)

plt.close(fig)

# ============================================================
# Downsample to exact 1920x1080
# ============================================================

img: Image.Image = Image.open(tmp_file).convert("RGB")

img = img.resize(
    (WIDTH, HEIGHT),
    Image.Resampling.LANCZOS,
)

# Optional extremely subtle sharpening
img = img.filter(
    ImageFilter.UnsharpMask(
        radius=0.5,
        percent=45,
        threshold=2,
    )
)

output_file: str = f"lorenz_fullhd_wallpaper_{VERSION}.png"

img.save(
    output_file,
    format="PNG",
    optimize=True,
)

print(f"Saved: {output_file}")
print(f"Resolution: {WIDTH}x{HEIGHT}")
print(f"Rotation: {ROTATION_ANGLE}°")