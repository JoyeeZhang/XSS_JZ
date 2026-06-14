import matplotlib
matplotlib.use('TkAgg')   # separate pop-up window

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# =================================================
# USER OPTION
# =================================================
# Choose one:
#   'center_rotation'    -> center moves on a circle
#   'center_translation' -> center only translates, while source/detector rotate around it
mode = 'center_translation'

# =================================================
# Parameters
# =================================================
c0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)

# -------- center-rotation mode parameters --------
Rc = 4.0      # center orbit radius
Rs = 1.5      # source orbit radius around moving center
Rd = 1.5      # detector orbit radius around moving center
rt = 0.8      # torus tube radius

omega_c = 1.0
phi_c = 0.0

# -------- common source/detector rotation --------
omega_s = 1.0
phi_s = 0.0

# -------- center-translation mode parameters -----
pitch_per_turn = 2.0   # translation distance per full source rotation

# -------- detector size --------------------------
det_width = 4.0
det_height = 3.0

# -------- display options ------------------------
show_torus = True
save_animation = False
save_filename = 'Rolical_scan.gif'

# =================================================
# Frame Q
# rows = u^T, v^T, w^T
# local orbit plane = span(u, v)
# local translation direction = w
# =================================================
u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
Q = np.vstack((u, v, w))

# =================================================
# Time samples
# =================================================
t = np.linspace(0, 2*np.pi*4, 700)
beta = omega_s * t + phi_s

# =================================================
# Geometry construction
# =================================================
if mode == 'center_rotation':
    alpha = omega_c * t + phi_c

    # Center rotates on a circle
    C_local = np.vstack((
        Rc * np.cos(alpha),
        Rc * np.sin(alpha),
        np.zeros_like(alpha)
    ))

    # Source rotates around moving center in plane perpendicular to center orbit
    S_local = np.vstack((
        (Rc + Rs * np.cos(beta)) * np.cos(alpha),
        (Rc + Rs * np.cos(beta)) * np.sin(alpha),
        Rs * np.sin(beta)
    ))

    # Detector on opposite side
    D_local = np.vstack((
        (Rc - Rd * np.cos(beta)) * np.cos(alpha),
        (Rc - Rd * np.cos(beta)) * np.sin(alpha),
        -Rd * np.sin(beta)
    ))

    # Detector frame vectors
    norm_local = np.vstack((
        np.cos(beta) * np.cos(alpha),
        np.cos(beta) * np.sin(alpha),
        np.sin(beta)
    ))

    channel_local = np.vstack((
        -np.sin(alpha),
         np.cos(alpha),
         np.zeros_like(alpha)
    ))

    row_local = np.vstack((
        -np.sin(beta) * np.cos(alpha),
        -np.sin(beta) * np.sin(alpha),
         np.cos(beta)
    ))

    # Center reference trajectory for plotting
    theta = np.linspace(0, 2*np.pi, 400)
    center_ref_local = np.vstack((
        Rc * np.cos(theta),
        Rc * np.sin(theta),
        np.zeros_like(theta)
    ))

elif mode == 'center_translation':
    # Center only translates along w
    h = pitch_per_turn * beta / (2.0 * np.pi)

    C_local = np.vstack((
        np.zeros_like(beta),
        np.zeros_like(beta),
        h
    ))

    # Source rotates around translating center
    S_local = np.vstack((
        Rs * np.cos(beta),
        Rs * np.sin(beta),
        h
    ))

    # Detector opposite source
    D_local = np.vstack((
        -Rd * np.cos(beta),
        -Rd * np.sin(beta),
        h
    ))

    # Detector frame vectors
    norm_local = np.vstack((
        np.cos(beta),
        np.sin(beta),
        np.zeros_like(beta)
    ))

    channel_local = np.vstack((
        -np.sin(beta),
         np.cos(beta),
         np.zeros_like(beta)
    ))

    row_local = np.vstack((
        np.zeros_like(beta),
        np.zeros_like(beta),
        np.ones_like(beta)
    ))

    # Center reference trajectory for plotting
    theta = np.linspace(0, np.max(beta), 400)
    h_ref = pitch_per_turn * theta / (2.0 * np.pi)
    center_ref_local = np.vstack((
        np.zeros_like(theta),
        np.zeros_like(theta),
        h_ref
    ))

else:
    raise ValueError("mode must be 'center_rotation' or 'center_translation'")

# =================================================
# Global coordinates
# =================================================
C = c0[:, None] + Q.T @ C_local
S = c0[:, None] + Q.T @ S_local
D = c0[:, None] + Q.T @ D_local

Cx, Cy, Cz = C
Sx, Sy, Sz = S
Dx, Dy, Dz = D

ve_norm = Q.T @ norm_local
ve_channel = Q.T @ channel_local
ve_row = Q.T @ row_local

# normalize for safety
ve_norm /= np.linalg.norm(ve_norm, axis=0, keepdims=True)
ve_channel /= np.linalg.norm(ve_channel, axis=0, keepdims=True)
ve_row /= np.linalg.norm(ve_row, axis=0, keepdims=True)

center_ref = c0[:, None] + Q.T @ center_ref_local

# =================================================
# Torus surface (shown only in center_rotation mode)
# =================================================
torus_exists = show_torus and (mode == 'center_rotation')
if torus_exists:
    tu = np.linspace(0, 2*np.pi, 100)
    tv = np.linspace(0, 2*np.pi, 60)
    TU, TV = np.meshgrid(tu, tv)

    Xt_local = (Rc + rt * np.cos(TV)) * np.cos(TU)
    Yt_local = (Rc + rt * np.cos(TV)) * np.sin(TU)
    Zt_local = rt * np.sin(TV)

    torus_local = np.stack([Xt_local, Yt_local, Zt_local], axis=0).reshape(3, -1)
    torus_global = c0[:, None] + Q.T @ torus_local

    Xt = torus_global[0].reshape(Xt_local.shape)
    Yt = torus_global[1].reshape(Yt_local.shape)
    Zt = torus_global[2].reshape(Zt_local.shape)

# =================================================
# Plot
# =================================================
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')

if torus_exists:
    ax.plot_surface(Xt, Yt, Zt, alpha=0.18, linewidth=0, antialiased=True)

ax.scatter([c0[0]], [c0[1]], [c0[2]], s=70, label='Fixed point')

# trajectories
ax.plot(center_ref[0], center_ref[1], center_ref[2],
        '--', linewidth=2, color='orange', label='Center trajectory')
ax.plot(Sx, Sy, Sz, color='red', alpha=0.35, label='Source trajectory')
# ax.plot(Dx, Dy, Dz, color='green', alpha=0.35, label='Detector center trajectory')

# animated objects
center_point, = ax.plot([], [], [], 'o', markersize=6, color='orange', label='Center')
source_point, = ax.plot([], [], [], 'o', markersize=8, color='red', label='Source')
detector_point, = ax.plot([], [], [], 's', markersize=7, color='green', label='Detector center')

line_CS, = ax.plot([], [], [], ':', linewidth=1.5, color='black')
line_CD, = ax.plot([], [], [], ':', linewidth=1.5, color='black')
line_SD, = ax.plot([], [], [], '--', linewidth=1.2, color='black')

source_trail, = ax.plot([], [], [], linewidth=2, color='red')
detector_trail, = ax.plot([], [], [], linewidth=2, color='green')

detector_panel, = ax.plot([], [], [], '-', linewidth=2, color='green', label='Detector panel')
det_axis_u, = ax.plot([], [], [], '-', linewidth=1.5, color='green')
det_axis_v, = ax.plot([], [], [], '-', linewidth=1.5, color='green')

# limits
xmin_candidates = [np.min(Sx), np.min(Dx)]
xmax_candidates = [np.max(Sx), np.max(Dx)]
ymin_candidates = [np.min(Sy), np.min(Dy)]
ymax_candidates = [np.max(Sy), np.max(Dy)]
zmin_candidates = [np.min(Sz), np.min(Dz)]
zmax_candidates = [np.max(Sz), np.max(Dz)]

if torus_exists:
    xmin_candidates.append(np.min(Xt))
    xmax_candidates.append(np.max(Xt))
    ymin_candidates.append(np.min(Yt))
    ymax_candidates.append(np.max(Yt))
    zmin_candidates.append(np.min(Zt))
    zmax_candidates.append(np.max(Zt))

xmin = min(xmin_candidates) - 2.0
xmax = max(xmax_candidates) + 2.0
ymin = min(ymin_candidates) - 2.0
ymax = max(ymax_candidates) + 2.0
zmin = min(zmin_candidates) - 2.0
zmax = max(zmax_candidates) + 2.0

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

if mode == 'center_rotation':
    ax.set_title('Center Rotation Mode')
else:
    ax.set_title('Center Translation Mode')

ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
ax.view_init(elev=25, azim=40)
ax.legend()

# =================================================
# Animation update
# =================================================
half_w = det_width / 2.0
half_h = det_height / 2.0

def update(i):
    # moving points
    center_point.set_data([Cx[i]], [Cy[i]])
    center_point.set_3d_properties([Cz[i]])

    source_point.set_data([Sx[i]], [Sy[i]])
    source_point.set_3d_properties([Sz[i]])

    detector_point.set_data([Dx[i]], [Dy[i]])
    detector_point.set_3d_properties([Dz[i]])

    # connecting lines
    line_CS.set_data([Cx[i], Sx[i]], [Cy[i], Sy[i]])
    line_CS.set_3d_properties([Cz[i], Sz[i]])

    line_CD.set_data([Cx[i], Dx[i]], [Cy[i], Dy[i]])
    line_CD.set_3d_properties([Cz[i], Dz[i]])

    line_SD.set_data([Sx[i], Dx[i]], [Sy[i], Dy[i]])
    line_SD.set_3d_properties([Sz[i], Dz[i]])

    # trails
    source_trail.set_data(Sx[:i+1], Sy[:i+1])
    source_trail.set_3d_properties(Sz[:i+1])

    # detector_trail.set_data(Dx[:i+1], Dy[:i+1])
    # detector_trail.set_3d_properties(Dz[:i+1])

    # detector panel
    Dk = D[:, i]
    uk = ve_channel[:, i]
    vk = ve_row[:, i]

    p00 = Dk - half_w * uk - half_h * vk
    p01 = Dk - half_w * uk + half_h * vk
    p11 = Dk + half_w * uk + half_h * vk
    p10 = Dk + half_w * uk - half_h * vk

    panel = np.column_stack((p00, p01, p11, p10, p00))
    detector_panel.set_data(panel[0], panel[1])
    detector_panel.set_3d_properties(panel[2])

    axis_u = np.column_stack((Dk - half_w * uk, Dk + half_w * uk))
    axis_v = np.column_stack((Dk - half_h * vk, Dk + half_h * vk))

    det_axis_u.set_data(axis_u[0], axis_u[1])
    det_axis_u.set_3d_properties(axis_u[2])

    det_axis_v.set_data(axis_v[0], axis_v[1])
    det_axis_v.set_3d_properties(axis_v[2])

    return (
        center_point, source_point, detector_point,
        line_CS, line_CD, line_SD,
        source_trail, detector_trail,
        detector_panel, det_axis_u, det_axis_v
    )

ani = FuncAnimation(fig, update, frames=len(t), interval=20, blit=False)

if save_animation:
    ani.save(save_filename, writer='pillow', fps=30)

plt.show()