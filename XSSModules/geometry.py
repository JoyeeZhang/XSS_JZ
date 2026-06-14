import numpy as np
import sys
import xml.etree.ElementTree as ET
import warnings
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import XSSModules.enum_list as em
import copy

'''============================================================================
Module Functions
============================================================================'''
'''----------------------------------------------------------------------------
[F] - normalize_vector()
[I] - v: 1D array vector to be normalized
[O] - v: 1D array vector normalized
[H] - Created by ZYu prior to 20250614
----------------------------------------------------------------------------'''
def normalize_vector(v):            
    try:
      norm_v = np.linalg.norm(v) # calculate the norm, default: two-norm
      v = v/norm_v
    except ZeroDivisionError:
      v = np.zeros(np.size(v))
      warnings.warn('Zero vector is not normalized!')

    return v
  
'''----------------------------------------------------------------------------
[F] - rotate_around_axis_RH(self,theta_deg,vaxis,vtarget):
[I] - theta_deg: rotation applied around vaxis following right hand rule
      can be either a scalar or a vector of N angles
[I] - vaxis: a 3 component array, vector of the rotating axis; 
      vaxis is normalized before ued, so don't have to be unit
[I] - vtarget: vector to be rotated 
      vtarget can be either a single 3 component array or 
      [N,3] 2D array that contains N 3 component target vectors
[O] - vrotated: vtarget rotated around vaxis folloing right hand rule
      vrotated can be either a single 3 component array or 
      [N,3] 2D array that contains N target 3 component arrays
[H] - Created by ZYu prior to 20250614
----------------------------------------------------------------------------''' 
def rotate_around_axis_RH(theta_deg,vtarget,vaxis=None):
    theta = np.deg2rad(theta_deg)
    vtarget = np.atleast_2d(vtarget)
    
    
    # Case of 2D vector
    if vtarget.shape[1] == 2:
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        rot_matrix = np.array = np.array([[costheta, -sintheta], [sintheta, costheta]])
        vrotated = vtarget @ rot_matrix.T
    # Case of 3D vector
    elif vtarget.shape[1] == 3:
        
        if vaxis is None:
            raise ValueError('vaxis must be provided for 3D rotation!')
        
        # normalized rotation axis
        ve_axis = normalize_vector(vaxis)
        costheta = np.outer(np.cos(theta),np.ones(3)) #[N,3]
        sintheta = np.outer(np.sin(theta),np.ones(3)) #[N,3]
                
        # target vector decomposed on the rotation axis direction
        vtgt1 = np.outer(np.dot(vtarget,ve_axis),ve_axis) # [N,3]
        vtgt2 = vtarget - vtgt1 #[N,3]
        vtgt3 = np.cross(ve_axis,vtgt2) #[N,3]
                
        vrotated = vtgt1 + vtgt2*costheta + vtgt3*sintheta
    
    else:
        raise ValueError('Target vector must be either a 2D or 3D!') 
    
    if vrotated.shape[0] == 1:
        vrotated = vrotated.flatten()
        
    return vrotated    
    
    
    # costheta = np.outer(np.cos(theta),np.ones(3)) #[N,3]
    # sintheta = np.outer(np.sin(theta),np.ones(3)) #[N,3]
            
    # # normalized rotation axis
    # ve_axis = normalize_vector(vaxis)
            
    # # target vector decomposed on the rotation axis direction
    # vtgt1 = np.outer(np.dot(vtarget,ve_axis),ve_axis) # [N,3]
    # vtgt2 = vtarget - vtgt1 #[N,3]
    # vtgt3 = np.cross(ve_axis,vtgt2) #[N,3]
            
    # vrotated = vtgt1 + vtgt2*costheta + vtgt3*sintheta
    
    # if vrotated.shape[0] == 1:
    #     vrotated = vrotated.flatten()
        
    # return vrotated

'''============================================================================
Class       
    Basis3D
Parent
    None
Attributes
    parent_crd - string: parent coordinate system
    vo  - 1D array 3: vector position of the origin in the parent coordinate sys
    ve0 - 1D array 3: 1st basis vector
    ve1 - 1D array 3: 2nd basis vector
    ve2 - 1D array 3: 3rd basis vector
Methods
    normalize_vector()
    rotate_around_axis_RH(theta_deg,vaxis,vtarget)
Notes
    - ve0,ve1,ve2 are orthonormal
Examples
    3D volume in the world coordinate system
        parent_crd = 'World'
        vo  - origin = [0,0,0]
        ve0 - ex = [1,0,0]
        ve1 - ey = [0,1,0]
        ve2 - ez = [0,0,1]
    2D projection in the detector coordiante system
        parent_crd = 'Xray Source'
        vo - detector origin = [0,0,-SDD]
        ve0 - eu
        ve1 - ev
        ve2 - ew
History: Created by ZYu prior to 20250614
============================================================================'''
class Basis3D():
    def __init__(self,
                 ve0 = np.array([1,0,0]),
                 ve1 = np.array([0,1,0]),
                 ve2 = np.array([0,0,1]),
                 vo  = np.array([0,0,0]), parent_crd = 'World'):
        
        self.ve0 = ve0 # basis vectors in the parental coordinate system
        self.ve1 = ve1
        self.ve2 = ve2
        
        self.vo  = vo  # Origin position in the parental coordinate system
        self.parent_crd = 'World'
        
        self.ve0 = normalize_vector(self.ve0)
        self.ve1 = normalize_vector(self.ve1)
        self.ve2 = normalize_vector(self.ve2)
        
'''============================================================================
Class       
    Basis2D
Parent
    None
Attributes
    parent_crd - string: parent coordinate system
    vo  - 1D array 3: vector position of the origin in the parent coordinate sys
    ve0 - 1D array 3: 1st basis vector
    ve1 - 1D array 3: 2nd basis vector
Methods
    normalize_vector()
    rotate_around_axis_RH(theta_deg,vaxis,vtarget)
Notes
    - ve0,ve1 are orthonormal
Examples
    2D volume in the world coordinate system
        parent_crd = 'World'
        vo  - origin = [0,0]
        ve0 - ex = [1,0]
        ve1 - ey = [0,1]
History: Created by ZYu prior to 20250614
============================================================================'''
class Basis2D():
    def __init__(self,
                 ve0 = np.array([1,0,0]),
                 ve1 = np.array([0,1,0]),
                 vo  = np.array([0,0,0]), parent_crd = 'World'):
        
        self.ve0 = ve0 # basis vectors in the parental coordinate system
        self.ve1 = ve1
        
        self.vo  = vo  # Origin position in the parental coordinate system
        self.parent_crd = 'World'
        
        self.ve0 = normalize_vector(self.ve0)
        self.ve1 = normalize_vector(self.ve1)
    
'''============================================================================
Class       
    Volume3D
Parent
    Basis3D
Attributes
    ve_col   - 1D array 3: 1st basis vector (e.g.,vex)
    ve_row   - 1D array 3: 2nd basis vector (e.g.,vey)
    ve_slice - 1D array 3: 3rd basis vector (e.g.,vez)
    ncol,dcol,ocol
    nrow,drow,orow
    nslice,dslice,oslice
    
    vo - 1D array 3: vector position of the origin in the parent coordinate sys
    parent_crd - string: parent coordinate system (default world)
Methods
    ---- Inherited
    normalize_vector()
    rotate_around_axis_RH(theta_deg,vaxis,vtarget):
    ---- Added
    normalize_basis
    rotate_around_axis0_RH(theta_deg)
    rotate_around_axis1_RH(theta_deg)
    rotate_around_axis2_RH(theta_deg)
Notes
    - ve_col,ve_row,ve_slice are orthonormal
    - a corresponding volume would be of size [nslice,nrow,ncol]
History: Created by ZYu prior to 20250614
============================================================================'''
class Volume3D():
    def __init__(self,
                 ncol,nrow,nslice,
                 dcol,drow,dslice,
                 ocol,orow,oslice,
                 ve_col = np.array([1,0,0]),
                 ve_row = np.array([0,1,0]),
                 ve_slice = np.array([0,0,1]),
                 vo  = np.array([0,0,0]), 
                 parent_crd = 'world'):
        
        self.ncol = ncol
        self.dcol = dcol
        self.ocol = ocol
        
        self.nrow = nrow
        self.drow = drow
        self.orow = orow
        
        self.nslice = nslice
        self.dslice = dslice
        self.oslice = oslice
        
        self.ve_col = ve_col # basis vectors in the parental coordinate system
        self.ve_row = ve_row
        self.ve_slice = ve_slice
        
        self.vo  = vo  # Origin position in the parental coordinate system
        self.parent_crd = 'World'
        
        self.normalize_basis()
    
    '''------------------------------------------------------------------------
    [F] - normalize_basis()
    [O] - basis vectors normalized
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def normalize_basis(self):
        self.ve_col = normalize_vector(self.ve_col)
        self.ve_row = normalize_vector(self.ve_row)
        self.ve_slice = normalize_vector(self.ve_slice)
    
    '''------------------------------------------------------------------------
    [F] - rotate_around_column_axis_RH(self,theta_deg): 
    [I] - theta_deg: rotation applied around the column axis following right hand rule
    [O] - ve_row and ve_slice are rotated by theta_deg
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------''' 
    def rotate_around_column_axis_RH(self,theta_deg): 
        self.ve_row = rotate_around_axis_RH(self,theta_deg,self.ve_row,self.ve_col)
        self.ve_slice = rotate_around_axis_RH(self,theta_deg,self.ve_slice,self.ve_col)
    
    '''------------------------------------------------------------------------
    [F] - rotate_around_row_axis_RH(self,theta_deg): 
    [I] - theta_deg: rotation applied around the row axis following right hand rule
    [O] - ve_col and ve_slice are rotated by theta_deg
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------''' 
    def rotate_around_row_axis_RH(self,theta_deg): 
        self.ve_slice = rotate_around_axis_RH(self,theta_deg,self.ve_slice,self.ve_row)
        self.ve_col = rotate_around_axis_RH(self,theta_deg,self.ve_col,self.ve_row)
        
    '''------------------------------------------------------------------------
    [F] - rotate_around_slice_axis_RH(self,theta_deg): 
    [I] - theta_deg: rotation applied around the row axis following right hand rule
    [O] - ve_col and ve_row are rotated by theta_deg
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------''' 
    def rotate_around_slice_axis_RH(self,theta_deg): 
        self.ve_col = rotate_around_axis_RH(self,theta_deg,self.ve_col,self.ve_slice)
        self.ve_row = rotate_around_axis_RH(self,theta_deg,self.ve_row,self.ve_slice)

'''============================================================================
Class       
    Slice2D
Parent
    Basis2D
Attributes
    ve_col   - 1D array 3: 1st basis vector (e.g.,vex)
    ve_row   - 1D array 3: 2nd basis vector (e.g.,vey)
    ncol,dcol,ocol
    nrow,drow,orow
    
    vo - 1D array 3: vector position of the origin in the parent coordinate sys
    parent_crd - string: parent coordinate system (default world)
Methods
    ---- Inherited
    normalize_vector()
    rotate_around_axis_RH(theta_deg,vaxis,vtarget):
Notes
    - ve_col and ve_row are orthonormal
    - a corresponding slice would be of size [nrow,ncol]
History: 20250429 created by ZYu
============================================================================'''
class Slice2D():
    def __init__(self,
                 ncol,nrow,
                 dcol,drow,
                 ocol,orow,
                 ve_col = np.array([1,0,0]),
                 ve_row = np.array([0,1,0]),
                 vo  = np.array([0,0,0]),
                 parent_crd = 'world'):
        
        self.ncol = ncol
        self.dcol = dcol
        self.ocol = ocol
        
        self.nrow = nrow
        self.drow = drow
        self.orow = orow
        
        self.ve_col = ve_col # basis vectors in the parental coordinate system
        self.ve_row = ve_row
        
        self.vo  = vo  # Origin position in the parental coordinate system
        self.parent_crd = 'World'
        
        self.normalize_basis()
    
    '''------------------------------------------------------------------------
    [F] - normalize_basis()
    [O] - basis vectors normalized
    [H] - 20250429 created by ZYu
    ------------------------------------------------------------------------'''
    def normalize_basis(self):
        self.ve_col = normalize_vector(self.ve_col)
        self.ve_row = normalize_vector(self.ve_row)
        
  
'''============================================================================
Class       
    CBFPD
Parent
    Basis3D
Attributes
    # direct input
    ve_channel
    ve_row
    ve_normal
    nchannel
    dchannel
    ochannel
    nrow
    drow
    orow
    SDD
    vsource
    
    # derived    
    SID - source2isocenter distance
    ve_source
    ochannel_piercing
    orow_piercing
    
Methods
    normalize_basis
    update_piercing_point
    update_detector_origin
    rotate_detector_around_channel_axis_at_piercing_RH
    rotate_detector_around_row_axis_at_piercing_RH
    rotate_detector_around_normal_axis_at_piercing_RH
    generate_divergent_cone_beam_unit_vector    
    
Notes
    
History: created by ZYu prior to 20250614
============================================================================'''
class CBFPD(Basis3D):
    def __init__(self, 
                 ve_channel = np.array([1, 0, 0]), # basis vector for channel index (veu)
                 ve_row = np.array([0, 0, 1]),     # basis vector for row index (vev)
                 ve_normal = np.array([0, -1, 0]),  # detector normal vector (vew)
                 nchannel  = 960,                  # number of channels
                 dchannel  = 0.45,                 # channel space (def in mm)
                 ochannel  = (960 - 1)/2,            # channel offset
                 nrow  = 960,                      # number of rows
                 drow  = 0.45,                     # row space (default in mm)
                 orow  = (960 - 1)/2,                # row offset
                 SDD = 1200,                       # source-to-detector distance
                 vsource  = 800*np.array([0, -1, 0])):           # assume in mm
        # Notes about the inputs: Removed SPD on April 26th, 2025
        # Detector coordinate system
        # - It is completely defined by ve_channel, ve_row, ve_normal, vsource & SDD
        # - Basis vectors: ve_channel, ve_row, ve_normal are orthonormal
        # - SDD is the source to detector distance
        # - The origin of the detector coordinate system is the orthogonal projection of the source onto the detector plane
        # SPD is the source-to-piercing point distance (the first version required to be provided, which is confusing)
        # - There is no need to provide SPD as it is derived from SDD and the angle between the source and the detector normal vector
        # - The piercing point is the intersection between the line connecting isocenter and source a & the detector plane
        self.vsource = vsource
        self.ve_channel = ve_channel
        self.ve_row = ve_row
        self.ve_normal = ve_normal
        self.nchannel = nchannel
        self.dchannel = dchannel
        self.ochannel = ochannel
        self.nrow = nrow
        self.drow = drow
        self.orow = orow 
        self.SDD = SDD
        
        self.normalize_basis()
        
        # ---- Derived parameters
        # source-to-axis distance
        self.SID = np.linalg.norm(self.vsource)
        self.ve_source = normalize_vector(self.vsource)
            
        # ---- Update source-to-piercing point related parameters
        self.ochannel_piercing = 0
        self.orow_piercing = 0
        self.SPD = [] # create the attribute to be updated later
        self.update_piercing_point()

    '''------------------------------------------------------------------------
    [F] - normalize_basis()
    [O] - basis vectors normalized
    [H] - created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def normalize_basis(self):
        self.ve_channel = normalize_vector(self.ve_channel)
        self.ve_row = normalize_vector(self.ve_row)
        self.ve_normal = normalize_vector(self.ve_normal)

    '''------------------------------------------------------------------------
    [F] - deep_copy()
    [O] - A deep copy (new instance) of the current object
    [H] - created by ZYu prior to 20250806
    ------------------------------------------------------------------------'''
    def deep_copy(self):
        return copy.deepcopy(self)
    
    '''------------------------------------------------------------------------
    [F] - update_piercing_point()
    [I] - The following attributes are used as inputs
          ve_source, vew, SDD
          veu, du, uoff
          vev, dv, voff       
    [O] - The following attributes are updated
          SPD, 
          uoff_piercing,
          voff_piercing]
    [N] - piercing point is the intersection between 
          the line connecting isocenter and source a &
          the detector plane
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def update_piercing_point(self):
         # SPD
        cos_vew_va = np.dot(self.ve_source, self.ve_normal)
        if np.abs(cos_vew_va) > sys.float_info.epsilon:
            self.SPD = self.SDD / cos_vew_va
        else:
            raise Exception('Unexpected geometry: Source is in the detector plane!!')
        
        # vector of piercing point relative to the orthogonal proj of va
        vd_piercing = self.SDD * self.ve_normal - self.SPD * self.ve_source
        upiercing = np.dot(vd_piercing,self.ve_channel)
        vpiercing = np.dot(vd_piercing,self.ve_row)
        
        self.ochannel_piercing = upiercing / self.dchannel + self.ochannel
        self.orow_piercing = vpiercing / self.drow + self.orow


    
    '''------------------------------------------------------------------------
    [F] - update_detector_origin()
    [I] - The following attributes are used as inputs
          ve_source, vew, SPD
          veu, du, uoff_piercing
          vev, dv, voff_piercing       
    [O] - The following attributes are updated
          SDD, 
          uoff,
          voff
    [N] - detector origin is the orthogonal projection of source onto the detector
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def update_detector_origin(self):
        # SDD
        cos_vew_va = np.dot(self.ve_source, self.ve_normal)
        self.SDD = self.SPD * cos_vew_va
        
        # uoff   
        cos_veu_va = np.dot(self.ve_source, self.ve_channel)
        self.ochannel = self.ochannel_piercing + self.SPD * cos_veu_va / self.dchannel
        
        # voff
        cos_vev_va = np.dot(self.ve_source,self.ve_row)
        self.orow = self.orow_piercing + self.SPD * cos_vev_va / self.drow

    '''------------------------------------------------------------------------
    [F] - rotate_around_channel_axis_RH(self,theta_deg): 
    [I] - theta_deg: rotation applied around the column axis following right hand rule
    [O] - vev and vew are rotated by theta_deg
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------''' 
    def rotate_detector_around_channel_axis_at_piercing_RH(self,theta_deg): 
        self.ve_row = rotate_around_axis_RH(theta_deg,self.ve_row,self.ve_channel)
        self.ve_normal = rotate_around_axis_RH(theta_deg,self.ve_normal,self.ve_channel)
        
        self.update_detector_origin()
    
    '''------------------------------------------------------------------------
    [F] - rotate_around_row_vaxis_RH(self,theta_deg): 
    [I] - theta_deg: rotation applied around the row axis following right hand rule
    [O] - veu and vew are rotated by theta_deg
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------''' 
    def rotate_detector_around_row_axis_at_piercing_RH(self,theta_deg): 
        self.ve_normal = rotate_around_axis_RH(theta_deg,self.ve_normal,self.ve_row)
        self.ve_channel = rotate_around_axis_RH(theta_deg,self.ve_channel,self.ve_row)
        
        self.update_detector_origin()
        
    '''------------------------------------------------------------------------
    [F] - rotate_around_normal_axis_RH(self,theta_deg): 
    [I] - theta_deg: rotation applied around the row axis following right hand rule
    [O] - veu and vev are rotated by theta_deg
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------''' 
    def rotate_detector_around_normal_axis_at_piercing_RH(self,theta_deg): 
        self.ve_channel = rotate_around_axis_RH(theta_deg,self.ve_channel,self.ve_normal)
        self.ve_row = rotate_around_axis_RH(theta_deg,self.ve_row,self.ve_normal)
        
        self.update_detector_origin()
    
    '''------------------------------------------------------------------------
    [F] - generate_divergent_cone_beam_unit_vector(self)
    [O] - valpha: 3xnuv matrix w/ each col being a unit vector for a det pix
    [N] - valpha = (u*veu + v*vev - D*vew)/sqrt(u*u+v*v+D*D)
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def generate_divergent_cone_beam_unit_vector(self):
        veu = self.ve_channel
        vev = self.ve_row
        vew = self.ve_normal
        D = self.SDD
        
        # Channel grids
        nu = self.nchannel
        du = self.dchannel        
        uoff = self.ochannel
        u = np.array([(i-uoff)*du for i in range(nu)])
        
        # Row grids
        nv = self.nrow
        dv = self.drow
        voff = self.orow
        v = np.array([(i-voff)*dv for i in range(nv)])
        
        # 2D grids
        uu = np.outer(np.ones(nv),u)
        vv = np.outer(v,np.ones(nu))
        
        # Calculate valpha: [3,nuv]
        nuv = nu*nv

        num = np.outer(veu,uu.flatten()) \
        + np.outer(vev,vv.flatten()) \
        - D*np.outer(vew,np.ones(nuv))
        
        den = np.sqrt((uu*uu).flatten() + (vv*vv).flatten() + D*D)

        valpha = num/den
        
        return valpha

'''============================================================================
Class       
    CBCurveD
Parent
    Basis3D
Attributes
    # direct input
    ve_channel
    ve_row
    ve_normal
    ve_gamma
    nchannel
    dchannel
    ochannel
    nrow
    drow
    orow
    dgamma
    SDD
    vsource
    
    # derived    
    SID - source2isocenter distance
    ve_source
    ochannel_piercing
    orow_piercing
    
Methods
    normalize_basis
    update_piercing_point
    update_detector_origin
    rotate_detector_around_channel_axis_at_piercing_RH
    rotate_detector_around_row_axis_at_piercing_RH
    rotate_detector_around_normal_axis_at_piercing_RH
    generate_divergent_cone_beam_unit_vector    
    return_distance_vector_origin
    
Notes
    
History: created by JZhang prior to 20250922
==============================================================================='''
class CBCurveD(Basis3D):
    def __init__(self,
                 ve_channel=np.array([1, 0, 0]),  # basis vector for channel index (veu)
                 ve_row=np.array([0, 0, 1]),  # basis vector for row index (vev)
                 ve_normal=np.array([0, -1, 0]),  # detector normal vector (vew)
                 nchannel=960,  # number of channels
                 dchannel=0.45,  # channel space (def in mm)
                 ochannel=(960 - 1) / 2,  # channel offset
                 nrow=960,  # number of rows
                 drow=0.45,  # row space (default in mm)
                 orow=(960 - 1) / 2,  # row offset
                 dgamma = 0.0625, # gamma interval (def in rad)
                 SDD = 1200,  # source-to-detector distance
                 vsource = 800 * np.array([0, -1, 0])):  # assume in mm
        # Notes about the inputs: Removed SPD on April 26th, 2025
        # Detector coordinate system
        # - It is completely defined by ve_channel, ve_row, ve_normal, vsource & SDD
        # - Basis vectors: ve_channel, ve_row, ve_normal are orthonormal
        # - SDD is the source to detector distance
        # - The origin of the detector coordinate system is the orthogonal projection of the source onto the detector plane
        # SPD is the source-to-piercing point distance (the first version required to be provided, which is confusing)
        # - There is no need to provide SPD as it is derived from SDD and the angle between the source and the detector normal vector
        # - The piercing point is the intersection between the line connecting isocenter and source a & the detector plane
        self.vsource = vsource
        self.ve_channel = ve_channel
        self.ve_row = ve_row
        self.ve_normal = ve_normal
        self.nchannel = nchannel
        self.dchannel = dchannel
        self.ochannel = ochannel
        self.nrow = nrow
        self.drow = drow
        self.orow = orow
        self.dgamma = dgamma
        self.SDD = SDD

        self.normalize_basis()

        # ---- Derived parameters
        # source-to-axis distance
        self.SID = np.linalg.norm(self.vsource)
        self.ve_source = normalize_vector(self.vsource)

        # ---- Update source-to-piercing point related parameters
        self.ochannel_piercing = 0
        self.orow_piercing = 0
        self.SPD = []  # create the attribute to be updated later
        self.update_piercing_point()

    '''------------------------------------------------------------------------
    [F] - normalize_basis()
    [O] - basis vectors normalized
    [H] - created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def normalize_basis(self):
        self.ve_channel = normalize_vector(self.ve_channel)
        self.ve_row = normalize_vector(self.ve_row)
        self.ve_normal = normalize_vector(self.ve_normal)

    '''------------------------------------------------------------------------
        [F] - deep_copy()
        [O] - A deep copy (new instance) of the current object
        [H] - created by ZYu prior to 20250806
        ------------------------------------------------------------------------'''

    def deep_copy(self):
        return copy.deepcopy(self)

    '''------------------------------------------------------------------------
    [F] - update_piercing_point()
    [I] - The following attributes are used as inputs
          ve_source, vew, SDD
          veu, du, uoff
          vev, dv, voff       
    [O] - The following attributes are updated
          SPD, 
          uoff_piercing,
          voff_piercing]
    [N] - piercing point is the intersection between 
          the line connecting isocenter and source a &
          the detector plane
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def update_piercing_point(self):
        # SPD
        cos_vew_va = np.dot(self.ve_source, self.ve_normal)
        if np.abs(cos_vew_va) > sys.float_info.epsilon:
            self.SPD = self.SDD / cos_vew_va
        else:
            raise Exception('Unexpected geometry: Source is in the detector plane!!')

        # vector of piercing point relative to the orthogonal proj of va
        vd_piercing = self.SDD * self.ve_normal - self.SPD * self.ve_source
        upiercing = np.dot(vd_piercing, self.ve_channel)
        vpiercing = np.dot(vd_piercing, self.ve_row)

        self.ochannel_piercing = upiercing / self.dchannel + self.ochannel
        self.orow_piercing = vpiercing / self.drow + self.orow

    '''------------------------------------------------------------------------
    [F] - update_detector_origin()
    [I] - The following attributes are used as inputs
          ve_source, vew, SPD
          veu, du, uoff_piercing
          vev, dv, voff_piercing       
    [O] - The following attributes are updated
          SDD, 
          uoff,
          voff
    [N] - detector origin is the orthogonal projection of source onto the detector
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def update_detector_origin(self):
        # SDD
        cos_vew_va = np.dot(self.ve_source, self.ve_normal)
        self.SDD = self.SPD * cos_vew_va

        # uoff
        cos_veu_va = np.dot(self.ve_source, self.ve_channel)
        self.ochannel = self.ochannel_piercing + self.SPD * cos_veu_va / self.dchannel

        # voff
        cos_vev_va = np.dot(self.ve_source, self.ve_row)
        self.orow = self.orow_piercing + self.SPD * cos_vev_va / self.drow

    '''------------------------------------------------------------------------
    [F] - rotate_around_channel_axis_RH(self,theta_deg): 
    [I] - theta_deg: rotation applied around the column axis following right hand rule
    [O] - vev and vew are rotated by theta_deg
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def rotate_detector_around_channel_axis_at_piercing_RH(self, theta_deg):
        self.ve_row = rotate_around_axis_RH(theta_deg, self.ve_row, self.ve_channel)
        self.ve_normal = rotate_around_axis_RH(theta_deg, self.ve_normal, self.ve_channel)

        self.update_detector_origin()

    '''------------------------------------------------------------------------
    [F] - rotate_around_row_vaxis_RH(self,theta_deg): 
    [I] - theta_deg: rotation applied around the row axis following right hand rule
    [O] - veu and vew are rotated by theta_deg
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def rotate_detector_around_row_axis_at_piercing_RH(self, theta_deg):
        self.ve_normal = rotate_around_axis_RH(theta_deg, self.ve_normal, self.ve_row)
        self.ve_channel = rotate_around_axis_RH(theta_deg, self.ve_channel, self.ve_row)

        self.update_detector_origin()

    '''------------------------------------------------------------------------
    [F] - rotate_around_normal_axis_RH(self,theta_deg): 
    [I] - theta_deg: rotation applied around the row axis following right hand rule
    [O] - veu and vev are rotated by theta_deg
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def rotate_detector_around_normal_axis_at_piercing_RH(self, theta_deg):
        self.ve_channel = rotate_around_axis_RH(theta_deg, self.ve_channel, self.ve_normal)
        self.ve_row = rotate_around_axis_RH(theta_deg, self.ve_row, self.ve_normal)

        self.update_detector_origin()

    '''------------------------------------------------------------------------
    [F] - generate_divergent_cone_beam_unit_vector(self)
    [O] - valpha: 3xnuv matrix w/ each col being a unit vector for a det pix
    [N] - valpha = (u*veu + v*vev - D*vew)/sqrt(u*u+v*v+D*D)
    [H] - Created by ZYu prior to 20250614
        - Revised by JZhang prior to 20260404
    ------------------------------------------------------------------------'''

    def generate_divergent_cone_beam_unit_vector(self):
        veu = self.ve_channel
        vev = self.ve_row
        vew = self.ve_normal
        D = self.SDD

        # Gamma grids
        ngamma = self.nchannel
        dgamma = self.dgamma
        gammaoff = self.ochannel
        gamma = np.array([np.deg2rad((i - gammaoff) * dgamma) for i in range(ngamma)])

        # Row grids
        nv = self.nrow
        dv = self.drow
        voff = self.orow
        v = np.array([(i - voff) * dv for i in range(nv)])

        # 2D grids
        vv = np.outer(v, np.ones(ngamma))
        gg = np.outer(np.ones(nv), gamma)

        # Calculate valpha: [3,nuv]
        num = D * np.outer(veu, np.sin(gg).flatten()) \
              + np.outer(vev, vv.flatten()) \
              - D * np.outer(vew, np.cos(gg).flatten())

        den = np.sqrt(D * D + (vv * vv).flatten())

        valpha = num / den

        return valpha

    '''------------------------------------------------------------------------
    [F] - return_distance_vector_origin(self)
    [O] - valpha: 3xnuv matrix w/ each col being a unit vector for a det pix
    [N] - valpha = (u*veu + v*vev - D*vew) / sqrt(u*u + v*v + D*D)
    [H] - Created by JZhang prior to 20250927
    ------------------------------------------------------------------------'''

    def return_distance_vector_origin(self):
        nv = self.nrow
        voff = self.orow
        nu = self.nchannel
        valpha = self.generate_divergent_cone_beam_unit_vector()
        vsource_array = np.vstack([self.vsource]*valpha.shape[1]).T
        v_dis = - vsource_array + np.sum(vsource_array*valpha, axis=0) * valpha
        v_dis[2, :] = 0.0
        dis = np.linalg.norm(v_dis[:2, :], axis=0)
        dis = dis.reshape(nv, nu)
        dis = dis[int(voff), :]

        return dis

    
'''============================================================================
Class       
    FBFPD
Parent
    Basis2D
Attributes
    # direct input
    ve_channel
    ve_normal
    nchannel
    dchannel
    ochannel
    SDD
    vsource
    
    # derived    
    SID - source2isocenter distance
    ve_source
    ochannel_piercing
    
Methods
    normalize_basis
    update_piercing_point
    update_detector_origin
    rotate_detector_around_channel_axis_at_piercing_RH
    rotate_detector_around_row_axis_at_piercing_RH
    rotate_detector_around_normal_axis_at_piercing_RH
    generate_divergent_cone_beam_unit_vector    
    
Notes
    
History: created by ZYu prior to 20250614
============================================================================'''
class FBFPD_2D(Basis2D):
    def __init__(self, 
                 ve_channel = np.array([0, 1, 0]), # basis vector for channel index (veu)
                 ve_normal = np.array([1, 0, 0]),  # detector normal vector (vew)
                 nchannel  = 432,                  # number of channels
                 dchannel  = 0.0,                 # channel space (def in mm)
                 ochannel  = 432/2+0.25,            # channel offset
                 SDD = 1200,                       # source-to-detector distance
                 vsource  = 800*np.array([1,0])):           # assume in mm            
        # Notes about the inputs: Removed SPD on April 26th, 2025
        # Detector coordinate system
        # - It is completely defined by ve_channel, ve_normal, vsource & SDD
        # - Basis vectors: ve_channel and ve_normal are orthonormal
        # - SDD is the source to detector distance
        # - The origin of the detector coordinate system is the orthogonal projection of the source onto the detector plane
        self.vsource = vsource
        self.ve_channel = ve_channel
        self.ve_normal = ve_normal
        self.nchannel = nchannel
        self.dchannel = dchannel
        self.ochannel = ochannel
        self.SDD = SDD
        
        self.normalize_basis()
        
        # ---- Derived parameters
        # source-to-axis distance
        self.SID = np.linalg.norm(self.vsource)
        self.ve_source = normalize_vector(self.vsource)
            
        # ---- Update source-to-piercing point related parameters
        self.ochannel_piercing = 0
        self.SPD = [] # create the attribute to be updated later
        self.update_piercing_point()
    '''------------------------------------------------------------------------
    [F] - normalize_basis()
    [O] - basis vectors normalized
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def normalize_basis(self):
        self.ve_channel = normalize_vector(self.ve_channel)
        self.ve_normal = normalize_vector(self.ve_normal)
    
    '''------------------------------------------------------------------------
    [F] - update_piercing_point()
    [I] - The following attributes are used as inputs
          ve_source, vew, SDD
          veu, du, uoff
          vev, dv, voff       
    [O] - The following attributes are updated
          SPD, 
          uoff_piercing,
          voff_piercing]
    [N] - piercing point is the intersection between 
          the line connecting isocenter and source a &
          the detector plane
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def update_piercing_point(self):
         # SPD
        cos_vew_va = np.dot(self.ve_source, self.ve_normal)
        if np.abs(cos_vew_va) > sys.float_info.epsilon:
            self.SPD = self.SDD/cos_vew_va
        else:
            raise Exception('Unexpected geometry: Source is in the detector plane!!')
        
        # vector of piercing point relative to the orthogonal proj of va
        vd_piercing = self.SDD*self.ve_normal - self.SPD*self.ve_source
        upiercing = np.dot(vd_piercing,self.ve_channel)
        
        self.ochannel_piercing = upiercing/self.dchannel + self.ochannel
    
    '''------------------------------------------------------------------------
    [F] - update_detector_origin()
    [I] - The following attributes are used as inputs
          ve_source, vew, SPD
          veu, du, uoff_piercing
          vev, dv, voff_piercing       
    [O] - The following attributes are updated
          SDD, 
          uoff,
          voff
    [N] - detector origin is the orthogonal projection of source onto the det
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def update_detector_origin(self):
        # SDD
        cos_vew_va = np.dot(self.ve_source, self.ve_normal)
        self.SDD = self.SPD*cos_vew_va
        
        # uoff   
        cos_veu_va = np.dot(self.ve_source,self.ve_channel)
        self.ochannel = self.ochannel_piercing + self.SPD*cos_veu_va/self.dchannel
    
    '''------------------------------------------------------------------------
    [F] - rotate_around_row_vaxis_RH(self,theta_deg): 
    [I] - theta_deg: rotation applied around the row axis following right hand rule
    [O] - veu and vew are rotated by theta_deg
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------''' 
    def rotate_detector_around_row_axis_at_piercing_RH(self,theta_deg): 
        self.ve_normal = rotate_around_axis_RH(theta_deg,self.ve_normal)
        self.ve_channel = rotate_around_axis_RH(theta_deg,self.ve_channel)
        
        self.update_detector_origin()
    
    '''------------------------------------------------------------------------
    [F] - generate_divergent_cone_beam_unit_vector(self)
    [O] - valpha: 3xnu matrix w/ each col being a unit vector for a det pix
    [N] - valpha = (u*veu - D*vew)/sqrt(u*u+D*D)
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def generate_divergent_cone_beam_unit_vector(self):
        veu = self.ve_channel
        vew = self.ve_normal
        D = self.SDD
        
        # Channel grids
        nu = self.nchannel
        du = self.dchannel        
        uoff = self.ochannel
        u = np.array([(i-uoff)*du for i in range(nu)])
        
        num = np.outer(veu,u) - D*np.outer(vew,np.ones(nu))
        
        den = np.sqrt(u*u + D*D)
        
        valpha = num/den
        
        return valpha


