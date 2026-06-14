import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math
matplotlib.use('TkAgg')

def generate_torus(xc, Q_rot, Torus_radius, Tube_radius):
    theta = np.linspace(0, 2*np.pi, 200)
    phi = np.linspace(0, 2*np.pi, 200)
    theta0, phi0 = np.meshgrid(theta, phi)

    x0 = (Torus_radius + Tube_radius * np.cos(theta0)) * np.cos(phi0)
    y0 = (Torus_radius + Tube_radius * np.cos(theta0)) * np.sin(phi0)
    z0 = Tube_radius * np.sin(theta0)

    pts = np.stack((x0, y0, z0), axis=0).reshape(3, -1)
    rotated = Q_rot.T @ pts + xc[:, None]
    x, y, z = rotated[0, :], rotated[1, :], rotated[2, :]
    x = x.reshape(x0.shape)
    y = y.reshape(y0.shape)
    z = z.reshape(z0.shape)

    return x, y, z

def generate_line(xa, alpha):
    t = np.linspace(0, 100, 100)
    pts = xa[:, None] + alpha[:, None] * t
    x, y, z = pts[0, :], pts[1, :], pts[2, :]

    return x, y, z

def generate_plane(x_norm, x0):
    n = np.array(x_norm, dtype=float)
    n = n / np.linalg.norm(n)

    if abs(n[0]) < 0.9:
        tmp = np.array([1, 0, 0], float)
    else:
        tmp = np.array([0, 1, 0], float)

    u1 = np.cross(n, tmp)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(n, u1)

    t = np.linspace(-8, 8, 200)
    s = np.linspace(-8, 8, 200)
    T, S = np.meshgrid(t, s)

    x = x0[0] + T*u1[0] + S*u2[0]
    y = x0[1] + T*u1[1] + S*u2[1]
    z = x0[2] + T*u1[2] + S*u2[2]

    return x, y, z


if __name__ == '__main__':
    Torus_r = 5.0
    Tube_r = 2.0
    x0 = np.array([0.0, 0.0, 0.0])

    va = np.array([1.0, 0.0, 0.0])
    vb = np.array([0.0, 1.0, 0.0])
    vc = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(va, vb)) > 1e-8 or abs(np.dot(va, vc)) or abs(np.dot(vb, vc)):
        raise ValueError("Vectors are not orthogonal")
    e_a = va / np.linalg.norm(va)
    e_b = vb / np.linalg.norm(vb)
    e_c = vc / np.linalg.norm(vc)
    Q_rot = np.stack((e_a, e_b, e_c), axis=-1)

    torus_x, torus_y, torus_z = generate_torus(x0, Q_rot, Torus_r, Tube_r)

    xa = np.array([0.0, -5.0, 0.0])
    valpha = np.array([0.0, 5.0, 10.0/np.sqrt(25-4)])
    alpha = valpha / np.linalg.norm(valpha)
    line_x, line_y, line_z = generate_line(xa, alpha)

    x_norm = np.array([0.0, 10.0/np.sqrt(25-4), 0.0])
    x0 = np.array([0.0, 0.0, 0.0])
    plane_x, plane_y, plane_z = generate_plane(x_norm, x0)

    x_norm1 = np.array([0.0, 5.0, 0.0])
    x01 = np.array([0.0, 0.0, 2.9])
    plane_x1, plane_y1, plane_z1 = generate_plane(x_norm1, x01)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter3D(*x0, color='purple', s=60)
    # ax.plot3D(line_x, line_y, line_z, color='red', linewidth=2)
    ax.plot_surface(torus_x, torus_y, torus_z, color='skyblue', edgecolor='k', alpha=0.4)
    # ax.plot_surface(plane_x, plane_y, plane_z, color='gray', edgecolor='none', alpha=0.4)
    # ax.plot_surface(plane_x1, plane_y1, plane_z1, color='yellow', edgecolor='none', alpha=0.4)

    # Plot the plane region inside torus
    inside_mask = (np.sqrt(plane_x ** 2 + plane_y ** 2) - Torus_r) ** 2 + plane_z ** 2 <= Tube_r ** 2
    plane_x_inside = np.where(inside_mask, plane_x, np.nan)
    plane_y_inside = np.where(inside_mask, plane_y, np.nan)
    plane_z_inside = np.where(inside_mask, plane_z, np.nan)
    # ax.plot_surface(plane_x_inside, plane_y_inside, plane_z_inside, color='red', edgecolor='none', alpha=1)

    '''
    # Plot the intersection of two planes
    inter_mask = abs((np.dot(x_norm1, x01) - x_norm1[0]*plane_x - x_norm1[1]*plane_y) / x_norm1[2] - plane_z) <= 0.05
    plane_x_inter = np.where(inter_mask, plane_x, np.nan)
    plane_y_inter = np.where(inter_mask, plane_y, np.nan)
    plane_z_inter = np.where(inter_mask, plane_z, np.nan)
    ax.plot3D(plane_x_inter, plane_y_inter, plane_z_inter, color='red', linewidth=2)
    '''

    ax_range = 1.2 * (Torus_r + Tube_r)
    ax.set_xlim(-ax_range, ax_range)
    ax.set_ylim(-ax_range, ax_range)
    ax.set_zlim(-ax_range, ax_range)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Torus Line Intersection')

    plt.show()

