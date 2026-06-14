import numpy as np
import os
import time
import matplotlib.pyplot as plt
from XSSModules import analytical_phantom as ap
from XSSModules import image as img
from XSSModules import recon as rcn
from XSSModules import geometry as geo
from XSSModules import file_io as fio

#=========================================================================
# Digital phantom
#=========================================================================
fn_phantom = 'FORBILD_head_static_LAC'
phantom = ap.StaticLACPhantom(fn_phantom)

dpixel = np.float64(0.6)
nx = 512
dx = dpixel
xoff= np.float64((nx-1)/2)
ny = 512
dy= dpixel
yoff= np.float64((ny-1)/2)
nz = 1
dz= dpixel
zoff= np.float64(0)

V = img.Volume3D(nx,ny,nz,dx,dy,dz,xoff,yoff,zoff)
V = phantom.generate_volume_image(V)
I = V.volume.squeeze()

fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(I, cmap='gray')
ax.axis('image')
plt.ion()
plt.pause(0.1)

#=========================================================================
# Generate analytical projections
#=========================================================================
# Define image
S = img.Slice2D(nx,ny,dx,dy,xoff,yoff)
S.slice = I
# Define FPD gridsc
R = 800
D = 1200
du= np.float64(0.9)
nu = 432
uoff= np.float64(216.25)
dv= np.float64(0.9)
nv = 1
voff= np.float64(0)

lambdas= np.float64(0)
nlambda = 720
dlambda= np.float64(2*np.pi/nlambda)

# fld_recon = 'C:\Workstation\Packages\XSS\XSSPackage\Data\FidelityFP\CTFBFPD_FS_MIC'
fld_recon = 'C:/Users/jzhan685/CT/Codes/XSS/Data/FidelityFP/CTFBFPD_FS_MIC'
os.makedirs(fld_recon, exist_ok=True)

header = fio.create_header_dict_projection_basic(nlambda,nv,nu,
                               dlambda,dv,du,
                               lambdas,voff,uoff,
                               data_type_numpy = np.float32,
                               projection_type_string = 'line_integral',
                               starting_axial_position = 0,
                               axial_movement_per_view = 0,
                               SDD = D,
                               SID = R)

#=========================================================================
# JHF
#=========================================================================
fnProjection = fn_phantom + 'I'+str(nx)+'_' + str(nu) + 'x' + str(nv) + 'x' + str(nlambda) + '_HF.prj'
fnProjection = os.path.join(fld_recon, fnProjection)
fio.write_header(fnProjection,header)
# gHF_cp= rcn.perform_FP_FB_Flat_Fidelity_GPU(S,lambdas,dlambda,nlambda,uoff, du, nu,R,D)
gHF_cp= rcn.perform_FP_FB_Flat_Fidelity(S,lambdas,dlambda,nlambda,uoff, du, nu,R,D)
gHF_np= gHF_cp.get()

fio.write_volume(fnProjection,gHF_np)

fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(gHF_np, cmap='gray')
ax.axis('image')
plt.ion()
plt.pause(0.01)
plt.show()

stoppoint = 1
