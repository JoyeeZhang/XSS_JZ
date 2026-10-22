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
Test_option = 'Geometry' # 'Geometry' or 'Projection' or None

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

# Test principal and oblique views over one complete rotation.
view_angles_deg = np.array([0.0, 30.0, 90.0, 137.0, 225.0, 315.0])


def rotate_around_y(vector, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    rotation = np.array([
        [cos_angle, 0.0, sin_angle],
        [0.0, 1.0, 0.0],
        [-sin_angle, 0.0, cos_angle],
    ])
    return rotation @ np.asarray(vector, dtype=np.float64)


def generate_view_vectors(angle_deg):
    veu_view = rotate_around_y(veu, angle_deg)
    vev_view = rotate_around_y(vev, angle_deg)
    vew_view = rotate_around_y(vew, angle_deg)
    source_view = R * vew_view
    return veu_view, vev_view, vew_view, source_view


def test_rotated_geometry(cone_beam_geometry, angle_deg):
    basis = np.vstack([
        cone_beam_geometry.ve_channel,
        cone_beam_geometry.ve_row,
        cone_beam_geometry.ve_normal,
    ])
    np.testing.assert_allclose(
        basis @ basis.T,
        np.eye(3),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.linalg.norm(cone_beam_geometry.vsource),
        R,
        rtol=1e-12,
        atol=1e-12,
    )

    ray_directions = cone_beam_geometry.generate_divergent_cone_beam_unit_vector()
    central_channel = int(round(cone_beam_geometry.ochannel))
    central_ray = ray_directions[:, central_channel]
    expected_central_ray = -cone_beam_geometry.vsource / R
    np.testing.assert_allclose(
        central_ray,
        expected_central_ray,
        rtol=1e-12,
        atol=1e-12,
        err_msg=f'central-ray geometry failed at {angle_deg} degrees',
    )


def calculate_expected_sphere_projection(cone_beam_geometry, circle_radius, circle_lac):
    ray_directions = cone_beam_geometry.generate_divergent_cone_beam_unit_vector()
    source_xz = cone_beam_geometry.vsource[[0, 2]]
    expected_projection = np.zeros(
        (cone_beam_geometry.nrow, cone_beam_geometry.nchannel),
        dtype=np.float64,
    )

    for ray in range(ray_directions.shape[1]):
        ray_direction_xz = ray_directions[[0, 2], ray]
        _, _, length = calculate_ray_circle_segment(
            source=source_xz,
            ray_dir=ray_direction_xz,
            circle_radius=circle_radius,
            circle_center=np.zeros(2, dtype=np.float64),
        )
        expected_projection.flat[ray] = length * circle_lac

    return expected_projection

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
    for view_angle_deg in view_angles_deg:
        veu_view, vev_view, vew_view, source_view = generate_view_vectors(
            view_angle_deg
        )
        geo_cbcurved = geo.CBCurveD(
            veu_view, vev_view, vew_view,
            nu, du, uoff, nv, dv, voff, dgamma, D, source_view,
        )
        test_rotated_geometry(geo_cbcurved, view_angle_deg)
        geo_cbcurved.test_curveD_geometry(1e-14, 1e-14)
        print(f'Geometry test passed at view {view_angle_deg:.1f} degrees')

if Test_option == 'Projection':

    circle_radius = 200.0
    circle_lac = 1.0
    circle_center = np.array([0.0, 0.0], dtype=np.float64)

    # -----------------------------------
    # Test for flat panel detector
    # -----------------------------------
    flat_maximum_error = 0.0
    for view_angle_deg in view_angles_deg:
        veu_view, vev_view, vew_view, source_view = generate_view_vectors(
            view_angle_deg
        )
        geo_cbfpd = geo.CBFPD(
            veu_view, vev_view, vew_view,
            nu, du, uoff, nv, dv, voff, D, source_view,
        )
        test_rotated_geometry(geo_cbfpd, view_angle_deg)
        projection = phantom.generate_line_integral_single_projection(geo_cbfpd)
        expected_projection = calculate_expected_sphere_projection(
            geo_cbfpd,
            circle_radius,
            circle_lac,
        )
        np.testing.assert_allclose(
            projection,
            expected_projection,
            rtol=1e-11,
            atol=1e-9,
            err_msg=f'flat-panel projection failed at {view_angle_deg} degrees',
        )
        maximum_error = np.max(np.abs(projection - expected_projection))
        flat_maximum_error = max(flat_maximum_error, maximum_error)
        print(
            f'Flat-panel projection passed at view {view_angle_deg:.1f} '
            f'degrees; maximum error={maximum_error:.3e}'
        )

    # -----------------------------------
    # Test for curved detector
    # -----------------------------------
    gamma = 30  # gamma in deg
    dgamma = gamma / nu
    ogamma = uoff
    curved_maximum_error = 0.0
    for view_angle_deg in view_angles_deg:
        veu_view, vev_view, vew_view, source_view = generate_view_vectors(
            view_angle_deg
        )
        geo_cbcurved = geo.CBCurveD(
            veu_view, vev_view, vew_view,
            nu, du, uoff, nv, dv, voff, dgamma, D, source_view,
        )
        test_rotated_geometry(geo_cbcurved, view_angle_deg)
        geo_cbcurved.test_curveD_geometry()
        projection = phantom.generate_line_integral_single_projection(geo_cbcurved)
        expected_projection = calculate_expected_sphere_projection(
            geo_cbcurved,
            circle_radius,
            circle_lac,
        )
        np.testing.assert_allclose(
            projection,
            expected_projection,
            rtol=1e-11,
            atol=1e-9,
            err_msg=f'curved-detector projection failed at {view_angle_deg} degrees',
        )
        maximum_error = np.max(np.abs(projection - expected_projection))
        curved_maximum_error = max(curved_maximum_error, maximum_error)
        print(
            f'Curved-detector projection passed at view {view_angle_deg:.1f} '
            f'degrees; maximum error={maximum_error:.3e}'
        )

    print(
        'All multi-view projection tests passed. '
        f'Flat maximum error={flat_maximum_error:.3e}; '
        f'curved maximum error={curved_maximum_error:.3e}'
    )
