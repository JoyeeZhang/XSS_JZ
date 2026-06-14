import numpy as np
import os
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
scale = 2
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
# Define FPD grids
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
fld_recon = './Data/FidelityFP/CTFBFPD_FS_MIC'
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
# Joseph - nups 1
#=========================================================================
nups = 1
print('Generating fanogram using average of upsampling rate:'+str(nups)+'...')
fnProjection = fn_phantom + 'I'+str(nx)+'_beamavg'+str(nups)+'x1'+'_' + str(nu) + 'x' + str(nv) + 'x' + str(nlambda) + '_Joseph.prj'
fnProjection = os.path.join(fld_recon, fnProjection)
fio.write_header(fnProjection,header)

gJ1 = rcn.perform_FP_FB_Flat_Joseph(S,lambdas,dlambda,nlambda,uoff, du, nu,R,D,nups)
fio.write_volume(fnProjection,gJ1)

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax1.imshow(gJ1, cmap='gray')
ax1.axis('off')
plt.pause(0.01)

#=========================================================================
# Joseph - nups 3
#=========================================================================
# nups = 3
# print('Generating fanogram using average of upsampling rate:'+str(nups)+'...')
# fnProjection = fn_phantom + 'I'+str(nx)+'_beamavg'+str(nups)+'x1'+'_' + str(nu) + 'x' + str(nv) + 'x' + str(nlambda) + '_Joseph.prj'
# fnProjection = os.path.join(fld_recon, fnProjection)
# fio.write_header(fnProjection,header)

# gJ3 = rcn.perform_FP_FB_Flat_Joseph(S,lambdas,dlambda,nlambda,uoff, du, nu,R,D,nups)
# fio.write_volume(fnProjection,gJ3)

# fig1 = plt.figure()
# ax1 = fig1.add_subplot()
# ax1.imshow(gJ3, cmap='gray')
# ax1.axis('off')
# plt.pause(0.01)

#=========================================================================
# Joseph - nups 5
#=========================================================================
# nups = 5
# print('Generating fanogram using average of upsampling rate:'+str(nups)+'...')
# fnProjection = fn_phantom + 'I'+str(nx)+'_beamavg'+str(nups)+'x1'+'_' + str(nu) + 'x' + str(nv) + 'x' + str(nlambda) + '_Joseph.prj'
# fnProjection = os.path.join(fld_recon, fnProjection)
# fio.write_header(fnProjection,header)

# gJ5 = rcn.perform_FP_FB_Flat_Joseph(S,lambdas,dlambda,nlambda,uoff, du, nu,R,D,nups)
# fio.write_volume(fnProjection,gJ5)

# fig1 = plt.figure()
# ax1 = fig1.add_subplot()
# ax1.imshow(gJ5, cmap='gray')
# ax1.axis('off')
# plt.pause(0.01)

#=========================================================================
# Joseph - nups 7
#=========================================================================
# nups = 7
# print('Generating fanogram using average of upsampling rate:'+str(nups)+'...')
# fnProjection = fn_phantom + 'I'+str(nx)+'_beamavg'+str(nups)+'x1'+'_' + str(nu) + 'x' + str(nv) + 'x' + str(nlambda) + '_Joseph.prj'
# fnProjection = os.path.join(fld_recon, fnProjection)
# fio.write_header(fnProjection,header)

# gJ7 = rcn.perform_FP_FB_Flat_Joseph(S,lambdas,dlambda,nlambda,uoff, du, nu,R,D,nups)
# fio.write_volume(fnProjection,gJ7)

# fig1 = plt.figure()
# ax1 = fig1.add_subplot()
# ax1.imshow(gJ7, cmap='gray')
# ax1.axis('off')
# plt.pause(0.01)

#=========================================================================
# Joseph - nups 9
#=========================================================================
# nups = 9
# print('Generating fanogram using average of upsampling rate:'+str(nups)+'...')
# fnProjection = fn_phantom + 'I'+str(nx)+'_beamavg'+str(nups)+'x1'+'_' + str(nu) + 'x' + str(nv) + 'x' + str(nlambda) + '_Joseph.prj'
# fnProjection = os.path.join(fld_recon, fnProjection)
# fio.write_header(fnProjection,header)

# gJ9 = rcn.perform_FP_FB_Flat_Joseph(S,lambdas,dlambda,nlambda,uoff, du, nu,R,D,nups)
# fio.write_volume(fnProjection,gJ9)

# fig1 = plt.figure()
# ax1 = fig1.add_subplot()
# ax1.imshow(gJ9, cmap='gray')
# ax1.axis('off')
# plt.pause(0.01)

# stoppoint = 1
