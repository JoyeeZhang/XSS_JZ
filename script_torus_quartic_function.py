import numpy as np

if __name__ == '__main__':
    # coefficients of quartic function with removing cubic term
    E = 266.5
    F = 0
    G = 6530.5625

    # coeffeicients of cubic function with removing quadratic term
    H = - (E ** 2) / 12 - G
    L = - (E ** 3) / 108 - (F ** 2) / 8 + E * G / 3

    delta_cubic = (L / 2) ** 2 + (H / 3) ** 3

    if delta_cubic > 0:
        alpha = np.cbrt(- L / 2 + np.sqrt(delta_cubic)) + np.cbrt(- L / 2 - np.sqrt(delta_cubic)) + E / 6
    else:
        theta = np.arccos((3 * L) / (2 * H) * np.sqrt(- 3 / H)) / 3
        alpha = 2 * np.sqrt(- H / 3) * np.cos(theta) + E / 6

    t_temp = np.zeros((4, 1))
    if 2 * alpha - E < 0:
        t_temp[0] = - np.inf
        t_temp[1] = - np.inf
        t_temp[2] = - np.inf
        t_temp[3] = - np.inf
    else:
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