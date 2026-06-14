# %%
import sys
import numpy as np
import matplotlib.pyplot as plt
from XSSModules import analytical_phantom as ap
from XSSModules import image as img
from XSSModules import support as spt

# phantom_name = "Ball_static_LAC"
# phantom_name = "Pelvis_static_LAC"
# phantom_name = "Breast_static_LAC"
# phantom_name = "Prostate_static_LAC"
# phantom_name = "FORBILD_head_static_LAC"
phantom_name = "Tire3D_H44-5X16-5R21_LAC"
# phantom_name = "Tire2D_H44-5X16-5R21"

# Load phantom definition
phantom = ap.StaticLACPhantom(phantom_name)

# generate 3D digital image
save_option = True # True or False
volume_option = 'Full' # 'Partial' or 'Full'
scale= 0.2
xFOV = 256.0/scale
zFOV = 256.0/scale

if volume_option == 'Full':
    ncol = 1024
    nrow = 1024
    nslice = 1
    dcol = xFOV / ncol
    drow = xFOV / nrow
    dslice = zFOV / nslice
    ocol = (ncol - 1) / 2
    orow = (nrow - 1) / 2
    oslice = (nslice - 1) / 2
    vol = img.Volume3D(ncol, nrow, nslice, dcol, drow, dslice, ocol, orow, oslice)
    vol = phantom.generate_volume_image(vol)
    # show
    spt.multi_section_viewer(vol.volume, ivmin=0.02, ivmax=0.03, isection='slice')
    # save
    if save_option == True:
        fn = './Data/' + phantom_name + '_Full_' + str(ncol) + 'x' + str(nrow) + 'x' + str(nslice) +'.img'
        vol.write_header(fn)
        vol.write_volume(fn)
elif volume_option == 'Partial':
    ncol = 3072
    nrow = 1
    nslice = 3072
    dcol = xFOV / ncol
    drow = xFOV / nrow
    dslice = zFOV / nslice
    ocol = (ncol - 1) / 2
    orow = (nrow - 1) / 2
    oslice = (nslice - 1) / 2
    scaled = 1.0 / 3.0
    vol = img.Volume3D(ncol, nrow, nslice, dcol, drow, dslice, ocol, orow, oslice)
    vol = phantom.generate_volume_image(vol)
    pvol_ncol = int(ncol * scaled)
    pvol_nrow = int(nrow)
    pvol_nslice = int(nslice * scaled)
    pvol_ocol = (pvol_ncol-1) / 2
    pvol_orow = (pvol_nrow-1) / 2
    pvol_oslice = (pvol_nslice-1) / 2
    pvolume = vol.volume[int(oslice-pvol_oslice):int(oslice+pvol_oslice+1), :, int(538.2-pvol_ocol):int(538.2+pvol_ocol+1)]
    pvol = vol
    pvol.ncol = pvol_ncol
    pvol.nrow = pvol_nrow
    pvol.nslice = pvol_nslice
    pvol.ocol = pvol_ocol
    pvol.orow = pvol_orow
    pvol.oslice = pvol_oslice
    pvol.volume = pvolume
    # show
    spt.multi_section_viewer(pvolume, ivmin=0.02, ivmax=0.03, isection='row')
    # Save
    if save_option == True:
        fn = './Data/' + phantom_name + '_Partial_' + str(pvol_ncol) + 'x' + str(pvol_nrow) + 'x' + str(pvol_nslice) +'.img'
        pvol.write_header(fn)
        pvol.write_volume(fn)




