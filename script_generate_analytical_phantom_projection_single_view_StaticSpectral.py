import numpy as np
from XSSModules import analytical_phantom as ap
from XSSModules import image as img
from XSSModules import support as spt
from XSSModules import geometry as geo
from XSSModules.xray_spectra import SpekCalc as spk
import matplotlib.pyplot as plt


# Load spectrum
fn_spectrum = '120kVp 10deg 1200Air 0.8Be 1Al 0Cu 0Sn 0W 0Ta 0Wa.spec';
SP = spk(fn_spectrum)
# SP.plot_spectrum('counts')


# Load phantom
phantom_name = "FORBILD_head_static_material"

# Load phantom definition
phantom = ap.StaticSpectralPhantom(phantom_name)

# Define FPD grids
rFOV = 128
R = 1000
D = 1500
uFOV = 1.1*D*rFOV*2/np.sqrt(R*R - rFOV*rFOV)
du = 5.0
nu = int(uFOV/du)
uoff = (nu-1)/2
dv = 5.0
nv = nu
voff = (nv-1)/2

# Define source and detector orientation
veu = np.array([0,-1,0])
vev = np.array([0,0,-1])
vew = np.array([-1,0,0])

# Create CBFPD
geo_cbfpd = geo.CBFPD(veu,vev,vew,nu,du,uoff,nv,dv,voff,D,R*vew)

source_spectrum = SP.spectrum_counts_per_kevmm2mAs_1m # Single projection angle
energy_kev = SP.energy_kev  # Single energy level in keV

source_spectrum = source_spectrum[20]
energy_kev = energy_kev[20]  # Single energy level in keV

bPoisson = False

g = phantom.generate_line_integral_single_projection_FPD_PCD_InvariantFluence(geo_cbfpd,source_spectrum,energy_kev)

spt.single_slice_viewer(g)
plt.show(block=True)






