import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math

from matplotlib.pyplot import xcorr

matplotlib.use('TkAgg')

class Torus():
    def __init__(self, R, r, xc, Q):
        self.Torus_radius = R
        self.Tube_radius = r
        self.center = xc
        self.mat_rot = Q

    def generate_torus_3D(self, ntheta=200, nphi=200):
        R = self.Torus_radius
        r = self.Tube_radius
        xc = self.center
        Q = self.mat_rot

        theta = np.linspace(0, 2 * np.pi, ntheta)
        phi = np.linspace(0, 2 * np.pi, nphi)
        theta0, phi0 = np.meshgrid(theta, phi)

        x0 = (R + r * np.cos(theta0)) * np.cos(phi0)
        y0 = (R + r * np.cos(theta0)) * np.sin(phi0)
        z0 = r * np.sin(theta0)

        pts = np.stack((x0, y0, z0), axis=0).reshape(3, -1)
        rotated = Q.T @ pts + xc[:, None]
        x, y, z = rotated[0, :], rotated[1, :], rotated[2, :]
        x = x.reshape(x0.shape)
        y = y.reshape(y0.shape)
        z = z.reshape(z0.shape)

        return x, y, z

    def generate_torus_cross_section(self, dis, ntheta=2000):
        R = self.Torus_radius
        r = self.Tube_radius
        d = dis

        theta = np.linspace(0, 2 * np.pi, ntheta)
        rho = R + r * np.cos(theta)
        z0 = r * np.sin(theta)
        mask = rho**2 >= d**2
        y0 = np.sqrt(rho[mask]**2 - d**2)

        y = np.concatenate([-y0, y0])
        z = np.concatenate([z0[mask], z0[mask]])

        return y, z

    def is_inside_torus(self, point):
        R = self.Torus_radius
        r = self.Tube_radius
        xc = self.center
        Q = self.mat_rot

        D = np.diag([1.0, 1.0, 0.0])
        M = Q.T @ D @ Q
        Eq = ((point-xc).T @ (point-xc) + R**2 - r**2)**2 - 4*(R**2)*((point-xc).T @ M @ (point-xc))

        if Eq <= 0:
            return True
        else:
            return False

class Ray():
    def __init__(self, xs, valpha):
        self.source = xs
        self.direction = valpha

    def generate_ray_3D(self, nt=100):
        xs = self.source
        valpha = self.direction

        t = np.linspace(0, nt, nt)
        pts = xs[:, None] + valpha[:, None] * t
        x, y, z = pts[0, :], pts[1, :], pts[2, :]

        return x, y, z

    def generate_ray_2D(self, nt=100):
        xs = self.source
        valpha = self.direction

        t = np.linspace(0, nt, nt)
        pts = xs[:, None] + valpha[:, None] * t
        y, z = pts[0, :], pts[1, :]

        return y, z


def intersection_torus_ray_with_axis_rotation(torus, ray):
    R = torus.Torus_radius
    r = torus.Tube_radius
    xc = torus.center
    Q = torus.mat_rot
    ea, eb, ec = Q[:, 0].T, Q[:, 1].T, Q[:, 2].T

    xs = ray.source
    valpha = ray.direction

    if np.linalg.norm (np.cross(ec, valpha)) < 1e-12:
        eu = ea
        ev = eb
        ew = ec
    else:
        vn = np.cross(ec, valpha)
        vn_per = np.cross(vn, ec)
        eu = vn / np.linalg.norm(vn)
        ev = vn_per / np.linalg.norm(vn_per)
        ew = ec

    xs_rot = np.stack(((xs-xc).T @ ev, (xs-xc).T @ ew), axis=-1)
    valpha_rot = np.stack((valpha.T @ ev, valpha.T @ ew), axis=-1)
    ray_rot = Ray(xs_rot, valpha_rot)
    d = (xs-xc).T @ eu
    M = np.array([[1, 0], [0, 0]])

    t_temp = np.zeros((4, 1))
    t_temp[0] = -np.inf
    t_temp[1] = -np.inf
    t_temp[2] = -np.inf
    t_temp[3] = -np.inf

    m = xs_rot.T @ valpha_rot
    n = xs_rot.T @ xs_rot + d**2 + R**2 - r**2
    g = valpha_rot.T @ M @ valpha_rot
    k = xs_rot.T @ M @ xs_rot + d**2
    l = xs_rot.T @ M @ valpha_rot

    # coefficients of quartic function with removing cubic term (x^4 + E*x^2 + F*x + G = 0)
    E = - 2*(m**2) + 2*n - 4*(R**2)*g
    F = 8*(R**2)*(m*g - l)
    G = (m**2 - n)**2 + 4*(R**2)*(2*m*l - k - (m**2)*g)

    if np.abs(F) < 1e-12: # x^4 + E*x^2 + G = 0
        delta_quartic = E**2 - 4*G
        if delta_quartic >= 0:
            t_temp_square1 = (- E + np.sqrt(delta_quartic)) / 2
            if t_temp_square1 >= 0:
                t_temp[0] = np.sqrt(t_temp_square1)
                t_temp[1] = - np.sqrt(t_temp_square1)
            t_temp_square2 = (- E - np.sqrt(delta_quartic)) / 2
            if t_temp_square2 >= 0:
                t_temp[2] = np.sqrt(t_temp_square2)
                t_temp[3] = - np.sqrt(t_temp_square2)
    else: # (x^2+alpha)^2 - [(2*alpha-E)x^2 - F*x + (alpha^2-G)] = 0
        # coefficients of cubic function with removing quadratic term (alpha^3 + H*alpha + L = 0)
        H = - (E ** 2) / 12 - G
        L = - (E ** 3) / 108 - (F ** 2) / 8 + E * G / 3

        delta_cubic = (L / 2) ** 2 + (H / 3) ** 3

        alpha_temp = np.zeros((3, 1))
        alpha_temp[0] = -np.inf
        alpha_temp[1] = -np.inf
        alpha_temp[2] = -np.inf
        if delta_cubic > 1e-12:
            alpha_temp[0] = np.cbrt(- L / 2 + np.sqrt(delta_cubic)) + np.cbrt(- L / 2 - np.sqrt(delta_cubic)) + E / 6
        elif np.abs(delta_cubic) <= 1e-12:
            if np.abs(H) < 1e-12:
                alpha_temp[0] = 0
            else:
                if np.abs(L - 2 * (- H/3)**(3/2)) < 1e-12:
                    alpha_temp[0] = np.sqrt(- H/3)
                    alpha_temp[1] = - 2 * np.sqrt(- H/3)
                elif np.abs(L + 2 * (- H/3)**(3/2)) < 1e-12:
                    alpha_temp[0] = - np.sqrt(- H/3)
                    alpha_temp[1] = 2 * np.sqrt(- H/3)
        elif delta_cubic < 1e-12:
            theta = np.arccos((3 * L) / (2 * H) * np.sqrt(- 3 / H))
            alpha_temp[0] = 2 * np.sqrt(- H / 3) * np.cos(theta / 3) + E / 6
            alpha_temp[1] = 2 * np.sqrt(- H / 3) * np.cos((theta+2*np.pi) / 3) + E / 6
            alpha_temp[2] = 2 * np.sqrt(- H / 3) * np.cos((theta+4*np.pi) / 3) + E / 6

        alpha = np.max(alpha_temp)

        if 2 * alpha - E > 0:
            delta_quadratic1 = - 2 * alpha - E - 2 * F / np.sqrt(2 * alpha - E)
            delta_quadratic2 = - 2 * alpha - E + 2 * F / np.sqrt(2 * alpha - E)

            if delta_quadratic1 >= 0:
                t_temp[0] = (np.sqrt(2 * alpha - E) - np.sqrt(delta_quadratic1)) / 2
                t_temp[1] = (np.sqrt(2 * alpha - E) + np.sqrt(delta_quadratic1)) / 2
            else:
                t_temp[0] = - np.inf
                t_temp[1] = - np.inf

            if delta_quadratic2 >= 0:
                t_temp[2] = (- np.sqrt(2 * alpha - E) - np.sqrt(delta_quadratic2)) / 2
                t_temp[3] = (- np.sqrt(2 * alpha - E) + np.sqrt(delta_quadratic2)) / 2
            else:
                t_temp[2] = - np.inf
                t_temp[3] = - np.inf

    t_temp0 = np.roots([1, 0, E, F, G])
    # print("Results by me: ", t_temp)
    # print("Results by numpy: ", t_temp0)
    t = t_temp - m
    t = t[t >= 0]
    t = np.unique(t)
    t.sort()

    return t, ray_rot, d

def intersection_torus_ray(torus, ray):
    R = torus.Torus_radius
    r = torus.Tube_radius
    xc = torus.center
    Q = torus.mat_rot

    xs = ray.source
    valpha = ray.direction

    D = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    M = (Q.transpose().dot(D)).dot(Q)

    t_temp = np.zeros((4, 1))
    t_temp[0] = -np.inf
    t_temp[1] = -np.inf
    t_temp[2] = -np.inf
    t_temp[3] = -np.inf

    m = (xs - xc).T @ valpha
    n = (xs - xc).T @ (xs - xc) + R**2 - r**2
    g = valpha.T @ M @ valpha
    k = (xs - xc).T @ M @ (xs - xc)
    l = (xs - xc).T @ M @ valpha

    # coefficients of quartic function with removing cubic term (x^4 + E*x^2 + F*x + G = 0)
    E = - 2*(m**2) + 2*n - 4*(R**2)*g
    F = 8*(R**2)*(m*g - l)
    G = (m**2 - n)**2 + 4*(R**2)*(2*m*l - k - (m**2)*g)

    if np.abs(F) < 1e-12: # x^4 + E*x^2 + G = 0
        delta_quartic = E**2 - 4*G
        if delta_quartic >= 0:
            t_temp_square1 = (- E + np.sqrt(delta_quartic)) / 2
            if t_temp_square1 >= 0:
                t_temp[0] = np.sqrt(t_temp_square1)
                t_temp[1] = - np.sqrt(t_temp_square1)
            t_temp_square2 = (- E - np.sqrt(delta_quartic)) / 2
            if t_temp_square2 >= 0:
                t_temp[2] = np.sqrt(t_temp_square2)
                t_temp[3] = - np.sqrt(t_temp_square2)
    else: # (x^2+alpha)^2 - [(2*alpha-E)x^2 - F*x + (alpha^2-G)] = 0
        # coefficients of cubic function with removing quadratic term (alpha^3 + H*alpha + L = 0)
        H = - (E ** 2) / 12 - G
        L = - (E ** 3) / 108 - (F ** 2) / 8 + E * G / 3

        delta_cubic = (L / 2) ** 2 + (H / 3) ** 3

        alpha_temp = np.zeros((3, 1))
        alpha_temp[0] = -np.inf
        alpha_temp[1] = -np.inf
        alpha_temp[2] = -np.inf
        if delta_cubic > 0:
            alpha_temp[0] = np.cbrt(- L / 2 + np.sqrt(delta_cubic)) + np.cbrt(- L / 2 - np.sqrt(delta_cubic)) + E / 6
        elif np.abs(delta_cubic) == 0:
            if H == 0:
                alpha_temp[0] = 0
            else:
                if np.abs(L - 2 * (- H/3)**(3/2)) < 1e-12:
                    alpha_temp[0] = np.sqrt(- H/3)
                    alpha_temp[1] = - 2 * np.sqrt(- H/3)
                elif np.abs(L + 2 * (- H/3)**(3/2)) < 1e-12:
                    alpha_temp[0] = - np.sqrt(- H/3)
                    alpha_temp[1] = 2 * np.sqrt(- H/3)
        elif delta_cubic < 0:
            theta = np.arccos((3 * L) / (2 * H) * np.sqrt(- 3 / H))
            alpha_temp[0] = 2 * np.sqrt(- H / 3) * np.cos(theta / 3) + E / 6
            alpha_temp[1] = 2 * np.sqrt(- H / 3) * np.cos((theta+2*np.pi) / 3) + E / 6
            alpha_temp[2] = 2 * np.sqrt(- H / 3) * np.cos((theta+4*np.pi) / 3) + E / 6

        alpha = np.max(alpha_temp)

        if 2 * alpha - E > 0:
            delta_quadratic1 = - 2 * alpha - E - 2 * F / np.sqrt(2 * alpha - E)
            delta_quadratic2 = - 2 * alpha - E + 2 * F / np.sqrt(2 * alpha - E)

            if delta_quadratic1 >= 0:
                t_temp[0] = (np.sqrt(2 * alpha - E) - np.sqrt(delta_quadratic1)) / 2
                t_temp[1] = (np.sqrt(2 * alpha - E) + np.sqrt(delta_quadratic1)) / 2
            else:
                t_temp[0] = - np.inf
                t_temp[1] = - np.inf

            if delta_quadratic2 >= 0:
                t_temp[2] = (- np.sqrt(2 * alpha - E) - np.sqrt(delta_quadratic2)) / 2
                t_temp[3] = (- np.sqrt(2 * alpha - E) + np.sqrt(delta_quadratic2)) / 2
            else:
                t_temp[2] = - np.inf
                t_temp[3] = - np.inf

    t_temp0 = np.roots([1, 0, E, F, G])
    print("Results by me: ", t_temp)
    print("Results by numpy: ", t_temp0)
    t = t_temp - m
    t = t[t >= 0]
    t = np.unique(t)
    t.sort()

    return t


if __name__ == '__main__':
    torus_radius = 5.0
    tube_radius = 2.0
    xc = np.array([3.0, 0.0, 0.0])
    va = np.array([3.0, 2.0, 1.0])
    vb = np.array([3.0, -4.0, -1.0])
    vc = np.cross(va, vb)
    if abs(np.dot(va, vb)) > 1e-12 or abs(np.dot(va, vc)) or abs(np.dot(vb, vc)):
        raise ValueError("Vectors are not orthogonal")
    e_a = va / np.linalg.norm(va)
    e_b = vb / np.linalg.norm(vb)
    e_c = vc / np.linalg.norm(vc)
    Q_rot = np.stack((e_a, e_b, e_c), axis=0)
    torus = Torus(torus_radius, tube_radius, xc, Q_rot)

    xs = np.array([7, -7.0, -2.0])
    if torus.is_inside_torus(xs) == True:
        raise ValueError("The ray source should not be inside torus")
    valpha = np.array([1.0, 2.0, 1.0])
    valpha = valpha / np.linalg.norm(valpha)
    ray = Ray(xs, valpha)

    # t_rot, ray_rot, d_rot = intersection_torus_ray_with_axis_rotation(torus, ray)
    x_torus0, y_torus0, z_torus0 = torus.generate_torus_3D(ntheta=200, nphi=200)
    x_ray0, y_ray0, z_ray0 = ray.generate_ray_3D(nt=100)
    # y_torus, z_torus = torus.generate_torus_cross_section(dis=d_rot, ntheta=200)
    # y_ray, z_ray = ray_rot.generate_ray_2D(nt=100)
    # inter_points_rot = ray_rot.source[:, None] + np.tile(t_rot, (2, 1)) * ray_rot.direction[:, None]

    t = intersection_torus_ray(torus, ray)
    inter_points = ray.source[:, None] + np.tile(t, (3, 1)) * ray.direction[:, None]

    # plot 2D torus cross-section and ray with intersection
    '''
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(111)
    ax1.plot(y_ray, z_ray, color='blue')
    ax1.plot(y_torus, z_torus, color='green')
    for i in range(inter_points_rot.shape[1]):
        ax1.plot(inter_points_rot[0, i], inter_points_rot[1, i], color='red', marker='*')

    ax1_range = 1.2 * (torus_radius + tube_radius)
    ax1.set_xlim(-ax1_range, ax1_range)
    ax1.set_ylim(-ax1_range, ax1_range)
    ax1.set_xlabel('Y')
    ax1.set_ylabel('Z')
    '''

    fig2 = plt.figure(figsize=(10, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot3D(x_ray0, y_ray0, z_ray0, color='blue')
    ax2.plot_surface(x_torus0, y_torus0, z_torus0, color='yellow', alpha=0.4)
    ax2.scatter3D(inter_points[0, :], inter_points[1, :], inter_points[2, :], color='red', marker='*')
    ax2.quiver(0, 0, 0, e_a[0], e_a[1], e_a[2], color='black')
    ax2.quiver(0, 0, 0, e_b[0], e_b[1], e_b[2], color='black')
    ax2.quiver(0, 0, 0, e_c[0], e_c[1], e_c[2], color='black')
    ax2_range = 1.2 * (torus_radius + tube_radius)
    ax2.set_xlim(-ax2_range, ax2_range)
    ax2.set_ylim(-ax2_range, ax2_range)
    ax2.set_zlim(-ax2_range, ax2_range)
    ax2.set_box_aspect((1, 1, 1))
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.show()



