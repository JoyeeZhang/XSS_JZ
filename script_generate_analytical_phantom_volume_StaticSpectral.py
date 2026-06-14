# %%
import sys
import numpy as np
import matplotlib.pyplot as plt
from XSSModules import analytical_phantom as ap
from XSSModules import image as img
from XSSModules import support as spt

phantom_name = "FORBILD_head_static_material"

# Load phantom definition
energy_kev = 70
phantom = ap.StaticSpectralPhantom(phantom_name,energy_kev)

# Generate 3D digital image
xFOV = 256
zFOV = 256
ncol = 512 
nrow = 512
nslice = 1
dcol = xFOV/ncol
drow = xFOV/nrow
dslice = zFOV/nslice
ocol = (ncol-1)/2
orow = (nrow-1)/2
oslice = (nslice-1)/2

vol = img.Volume3D(ncol,nrow,nslice,dcol,drow,dslice,ocol,orow,oslice)
vol = phantom.generate_volume_image(vol) 
spt.multi_slice_viewer(vol.volume*50000,ivmin=900,ivmax=1100)

# Save
fn = '../Data/' + phantom_name + '_' + str(energy_kev) + 'kev_'+ str(ncol) + 'x' + str(nrow) + 'x' + str(nslice) +'.img'
vol.write_header(fn)
vol.write_volume(fn)


