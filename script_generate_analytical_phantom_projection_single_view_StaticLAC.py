import os
import numpy as np
from XSSModules import analytical_phantom as ap
from XSSModules import image as img
from XSSModules import support as spt
from XSSModules import geometry as geo
import matplotlib.pyplot as plt

# choose phantom
phantom_name = "Tire3D_H44-5X16-5R21_LAC"
# Load phantom definition
phantom = ap.StaticLACPhantom(phantom_name)

# Choose detector
Detector_option = 'FPD' # 'FPD' or 'CurveD'

# Define grids
rFOV = 300
R = 2000
D = 3000
uFOV = 1.1*D*rFOV*2/np.sqrt(R*R - rFOV*rFOV)
du = 0.4
nu = int(uFOV/du)
uoff = (nu-1)/2
dv = 0.4
nv = 5
voff = (nv-1)/2

# Define source and detector orientation
veu = np.array([0, 0, -1])
vev = np.array([0, 1, 0])
vew = np.array([1, 0, 0])

if Detector_option == 'FPD':
    # Create CBFPD
    geo_cbfpd = geo.CBFPD(veu, vev, vew, nu, du, uoff, nv, dv, voff, D, R*vew)
    g = phantom.generate_line_integral_single_projection(geo_cbfpd)
    spt.single_slice_viewer(g)
elif Detector_option == 'CurveD':
    # Create CBCD
    gamma = 20 # gamma in deg
    dgamma = gamma/nu
    geo_cbcurved = geo.CBCurveD(veu, vev, vew, nu, du, uoff, nv, dv, voff, dgamma, D, R*vew)
    g = phantom.generate_line_integral_single_projection(geo_cbcurved)
    spt.single_slice_viewer(g)


# ----------------------------------------------------------------------------
# CBCD forward projection test
# ----------------------------------------------------------------------------
'''
gamma_array = np.array([np.deg2rad((i - ogamma) * dgamma) for i in range(ngamma)])
true_dis = np.abs(R * np.sin(gamma_array))
cbcd_dis = geo_cbcd.return_distance_vector_origin()
print(np.array_equal(np.round(true_dis,6), np.round(cbcd_dis,6)))
'''







