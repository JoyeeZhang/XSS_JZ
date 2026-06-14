import numpy as np
import os
from XSSModules import analytical_phantom as ap
from XSSModules import image as img
from XSSModules import support as spt
from XSSModules import geometry as geo
import matplotlib.pyplot as plt
from XSSModules import scanner as sc
from XSSModules import file_io as fio
from XSSModules import recon as rcn
from XSSModules import image as img

# Prep for the digital phantom
fn_phantom = 'FORBILD_head_static_LAC'
phantom = ap.StaticLACPhantom(fn_phantom)
alpha = 0.7

nscale = 8# 1,2,4,8

# Image definition
dpixel = np.float64(2.4/nscale)
nx = 128*nscale
dx = dpixel
xoff= np.float64((nx-1)/2)
ny = 128*nscale
dy= dpixel
yoff= np.float64((ny-1)/2)
nz = 1
dz= dpixel
zoff= np.float64(0)

nslice = 1
dslice = 1
oslice = 0

# Recon folders
# fld_recon = 'C:\Workstation\Packages\XSS\XSSPackage\Data\FidelityFP\CTFBFPD_FS_MIC\FBP_FB_FPD_Alpha'+str(alpha)+'M'+str(nx)
fld_recon = './Data/FBP_FB_FPD_Alpha'+str(alpha)+'M'+str(nx)
os.makedirs(fld_recon, exist_ok=True)
img_header = fio.create_header_dict_volume_basic(nslice,ny,nx,dslice,dy,dx,oslice,yoff,xoff)

# =========================================================================
# Ground truth
fn_recon = fn_phantom+'_GroundTruth_'+str(ny)+'x'+str(nx)+'.img'
fn_recon = os.path.join(fld_recon,fn_recon)
V = img.Volume3D(nx,ny,nz,dx,dy,dz,xoff,yoff,zoff)
V = phantom.generate_volume_image(V)
I = V.volume.squeeze()

fio.write_header(fn_recon,img_header)
fio.append_slices(fn_recon,I)
'''
# =========================================================================
# Recon of the HF
fn_raw_data_path = "./Data/FidelityFP/CTFBFPD_FS_MIC/FORBILD_head_static_LACI"+str(nx)+"_432x1x720_HF.prj"
fn_recon = fn_phantom+'_HF_'+str(ny)+'x'+str(nx)+'.img'
fn_recon = os.path.join(fld_recon,fn_recon)
slice = img.Slice2D(nx, ny, dx, dy, xoff, yoff)

fano_header = fio.read_header(fn_raw_data_path)
lambdai = fano_header['starting_angular_position']
dlambda = fano_header['angular_space_per_view']
uoff = fano_header['ochannel']
du = fano_header['dchannel']
R = fano_header['SID']
D = fano_header['SDD']

g = fio.read_volume(fn_raw_data_path) #[nlambda,nu]
g = g.squeeze()
rcn.perform_FBFBP_Flat_FS_Ramp(slice,alpha,g,R,D,lambdai,dlambda,uoff,du,fn_recon)     

'''
# =========================================================================
# Recon of the Joseph - beamavg1
fn_raw_data_path = "./Data/FidelityFP/CTFBFPD_FS_MIC/FORBILD_head_static_LACI"+str(nx)+"_beamavg1x1_432x1x720_Joseph.prj"
fn_recon = fn_phantom+'_Joseph_BeamAvg1_'+str(ny)+'x'+str(nx)+'.img'
fn_recon = os.path.join(fld_recon,fn_recon)
slice = img.Slice2D(nx, ny, dx, dy, xoff, yoff)

fano_header = fio.read_header(fn_raw_data_path)
lambdai = fano_header['starting_angular_position']
dlambda = fano_header['angular_space_per_view']
uoff = fano_header['ochannel']
du = fano_header['dchannel']
R = fano_header['SID']
D = fano_header['SDD']

g = fio.read_volume(fn_raw_data_path) #[nlambda,nu]
g = g.squeeze()
rcn.perform_FBFBP_Flat_FS_Ramp(slice,alpha,g,R,D,lambdai,dlambda,uoff,du,fn_recon)   




            

