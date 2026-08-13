import os
import numpy as np
from XSSModules import analytical_phantom as ap
from XSSModules import image as img
from XSSModules import support as spt
from XSSModules import geometry as geo
import matplotlib.pyplot as plt

# Choose phantom
phantom_name = "Sphere_test_LAC" # sphere with radius of 200
# Load phantom definition
phantom = ap.StaticLACPhantom(phantom_name)

# Choose test type
Test_option = 'Projection' # 'Geometry' or 'Projection' or None

# Define grids
rFOV = 250
R = 2000
D = 3000
uFOV = 1.1*D*rFOV*2 / np.sqrt(R*R - rFOV*rFOV)
du = 1.0
nu = int(uFOV/du)
uoff = (nu-1)/2
dv = 1.0
nv = 1
voff = (nv-1)/2

# Define source and detector orientation
veu = np.array([0, 0, -1])
vev = np.array([0, 1, 0])
vew = np.array([1, 0, 0])

# Define the function to calculate segment between ray and circle
def calculate_ray_circle_segment(source, ray_dir, circle_radius, circle_center=None):
    source = np.asarray(source, dtype=np.float64)
    ray_dir = np.asarray(ray_dir, dtype=np.float64)
    if circle_center is None:
        circle_center = np.zeros(2, dtype=np.float64)
    else:
        circle_center = np.asarray(circle_center, dtype=np.float64)
    ray_norm = np.linalg.norm(ray_dir)
    if ray_norm == 0:
        raise ValueError("ray_dir must be nonzero.")

    ray_dir = ray_dir / ray_norm
    # Vector from source to circle center
    source_to_center = circle_center - source
    # Distance from the source to the closest point on the ray line
    t_closest = np.dot(source_to_center, ray_dir)
    closest_point = source + t_closest * ray_dir
    center_to_ray = circle_center - closest_point
    perpendicular_distance_squared = np.dot(center_to_ray, center_to_ray)
    radius_squared = circle_radius ** 2

    if perpendicular_distance_squared > radius_squared:
        return None, None, 0.0

    half_chord_length = np.sqrt(max(radius_squared - perpendicular_distance_squared, 0.0))
    t1 = t_closest - half_chord_length
    t2 = t_closest + half_chord_length

    if t2 <= 0:
        return None, None, 0.0

    t1 = max(t1, 0.0)
    p1 = source + t1 * ray_dir
    p2 = source + t2 * ray_dir
    length = t2 - t1

    return p1, p2, length

if Test_option == 'Geometry':
    gamma = 30
    dgamma = gamma/nu
    ogamma = uoff
    ngamma = nu
    geo_cbcurved = geo.CBCurveD(veu, vev, vew, nu, du, uoff, nv, dv, voff, dgamma, D, R*vew)
    test_result = geo_cbcurved.test_curveD_geometry()
    if test_result == True:
        print('Geometry test result is True')
    else:
        print('Geometry test Failed')
if Test_option == 'Projection':

    circle_radius = 200.0
    circle_lac = 1.0
    circle_center = np.array([0.0, 0.0], dtype=np.float64)

    # -----------------------------------
    # Test for flat panel detector
    # -----------------------------------
    geo_cbfpd = geo.CBFPD(veu, vev, vew, nu, du, uoff, nv, dv, voff, D, R * vew)
    projection = phantom.generate_line_integral_single_projection(geo_cbfpd)

    source = np.array([-R, 0.0])
    detector_center = np.array([D - R, 0.0])  # (1000, 0)

    segments = []
    expected_projection = np.zeros((1, nu), dtype=np.float64)

    for i in range(nu):
        u = i - (nu - 1) / 2.0

        detector_point = detector_center + np.array([0.0, u], dtype=np.float64)

        ray_vec = detector_point - source
        ray_dir = ray_vec / np.linalg.norm(ray_vec)

        p1, p2, length = calculate_ray_circle_segment(source = source,
                                                      ray_dir = ray_dir,
                                                      circle_radius = circle_radius,
                                                      circle_center = circle_center)

        expected_projection[0, i] = length * circle_lac

    print("Test result for flat-panel detector is:", np.allclose(projection, expected_projection, atol=1e-16))

    # -----------------------------------
    # Test for curved detector
    # -----------------------------------
    gamma = 30  # gamma in deg
    dgamma = gamma / nu
    ogamma = uoff
    geo_cbcurved = geo.CBCurveD(veu, vev, vew, nu, du, uoff, nv, dv, voff, dgamma, D, R * vew)
    projection = phantom.generate_line_integral_single_projection(geo_cbcurved)

    gamma_rads = np.array([np.deg2rad((i - ogamma)*dgamma) for i in range(nu)])

    segments = []
    expected_projection = np.zeros((1, nu), dtype=np.float64)

    for i, igamma_rad in enumerate(gamma_rads):

        # Ray direction for curved detector
        ray_dir = np.array([
            np.cos(igamma_rad),
            np.sin(igamma_rad)
        ])

        detector_point = source + D * ray_dir

        p1, p2, length = calculate_ray_circle_segment(source = source,
                                                      ray_dir = ray_dir,
                                                      circle_radius = circle_radius,
                                                      circle_center = circle_center)

        expected_projection[0, i] = length * circle_lac

    print("Test result for curved detector is:", np.allclose(projection, expected_projection, atol=1e-16))


