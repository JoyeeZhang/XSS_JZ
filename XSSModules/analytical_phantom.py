import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
import sys
from XSSModules import geometry as geo
from XSSModules import enum_list as enm
from XSSModules.cross_section import XcomCrossSection as cs

# Jiaying's path
os.environ['XSSLibraries'] = 'C:/Users/jzhan685/CT/Codes/XSS/XSSLibraries/'

# Adding Libraray and Phantom Database
path_XSSLibraries = os.environ['XSSLibraries']
path_AnalyticalPhantoms = os.path.join(path_XSSLibraries,'AnalyticalPhantoms')
sys.path.append(path_AnalyticalPhantoms)

TMAX = 1e10 #np.finfo(float).max
EPS = np.finfo(float).eps

'''============================================================================
Class       
    QuadraticShape3D
Parent
    geometry: Basis3D
Attributes
    parent_crd - string: parent coordinate system
    vo  - 1D array 3: vector position of the origin in the parent coordinate sys
    ve0 - 1D array 3: 1st basis vector
    ve1 - 1D array 3: 2nd basis vector
    ve2 - 1D array 3: 3rd basis vector
Methods
    normalize_basis()
    translate(vtr)
    rotate_around_axis0_RH(theta_deg)
    rotate_around_axis1_RH(theta_deg)
    rotate_around_axis2_RH(theta_deg)
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
         Revised by JZhang prior to 20260404
============================================================================'''
class QuadraticShape3D(geo.Basis3D):
    def __init__(self, shape = enm.QuadricShape['SPHERE'], LAC = 0, bprecedence = 0,
                 vo = np.array([0,0,0]),
                 vradii = np.array([1,1,1]),
                 ve0 = np.array([1,0,0]),
                 ve1 = np.array([0,1,0]),
                 ve2 = np.array([0,0,1]),
                 clip_matrix = np.zeros([0,4]),
                 parent_crd = 'World'):
        super().__init__(ve0, ve1, ve2, vo, parent_crd)
        self.shape = shape # shape: QuadricShape
        self.vradii = vradii  # vradii = np.array([ra,rb,rc])
        # clipping planes
        # --[d,ve_normal] with ve_normal being a 3-component unit row vector
        # --(vx - vo)'*ve <= d are retained > d are clipped
        # --To add definition of 1 clipping plane
        # --matrix_clip = np.vstack([matrix_clip,array])
        self.clip_matrix = clip_matrix
        self.LAC = LAC
        self.bprecedence = bprecedence # Precedence: 0 additive [default] and 1 precedence
        
    '''------------------------------------------------------------------------
    [F] - calculate_shape_characteristics(self)
    [O] - shape_cha = {
            'center': vx0,        # 1x3 array
            'transform_matrix': M,# 3x3 array
            'equation_value': EQ} # scalar
    [N] - (vx - vx0)'M(vx - vx0) = EQ
    [H] - Created by ZYu prior to 20250614
        - Revised by JZhang prior to 20260404
        {Add torus as a new 3D quadratic shape}
    ------------------------------------------------------------------------'''
    def calculate_shape_characteristics(self):
        shape = self.shape
        vx0 = self.vo
        vea = self.ve0
        veb = self.ve1
        vec = self.ve2
        ra = self.vradii[0]
        rb = self.vradii[1]
        rc = self.vradii[2]
        
        match shape.value:
            case enm.QuadricShape.SPHERE.value:
                M  = np.identity(3)
                EQ = ra*ra
            case enm.QuadricShape.ELLIPSOID.value:
                Q  = np.vstack([vea,veb,vec])
                S  = np.diag([1/(ra*ra), 1/(rb*rb), 1/(rc*rc)])
                M  = (Q.transpose().dot(S)).dot(Q)
                EQ = np.float64(1)
            case enm.QuadricShape.CYLINDER.value:
                Q  = np.vstack([vea,veb])
                S  = np.diag([1/(ra*ra), 1/(rb*rb)])
                M  = (Q.transpose().dot(S)).dot(Q)
                EQ = np.float64(1)
            case enm.QuadricShape.CONE.value:
                Q  = np.vstack([vea,veb,vec])
                S  = np.diag([1/(ra*ra), 1/(rb*rb), -1/(rc*rc)])
                M  = (Q.transpose().dot(S)).dot(Q)
                EQ = np.float64(0)
            case enm.QuadricShape.HYPERBOLOID1.value:
                Q  = np.vstack([vea,veb,vec])
                S  = np.diag([1/(ra*ra), 1/(rb*rb), -1/(rc*rc)])
                M  = (Q.transpose().dot(S)).dot(Q)
                EQ = np.float64(1)
            case enm.QuadricShape.HYPERBOLOID2.value:
                Q  = np.vstack([vea,veb,vec])
                S  = np.diag([1/(ra*ra), 1/(rb*rb), -1/(rc*rc)])
                M  = (Q.transpose().dot(S)).dot(Q)
                EQ = np.float64(-1)
            case enm.QuadricShape.TORUS.value:
                Q = np.vstack([vea, veb, vec])
                D = np.diag([1, 1, 0])
                M = (Q.transpose().dot(D)).dot(Q)
                EQ = np.float64(0)
            case _:
                print('Invalid object type!!')
        
        # Shape surface vx is defined by the following equiation
        # (vx - vx0)'M(vx - vx0) = EQ
        shape_cha = {'center': vx0,        # 1x3 array
                     'transform_matrix': M,# 3x3 array
                     'equation_value': EQ} # scalar
        
        return shape_cha

'''============================================================================
Class       
    StaticLACPhantom: 
        No motion
        LAC direclty specified
Parent
    None
Attributes
    phantom_type                static
    attenuation_type            LAC
    num_obj_precedence          int
    num_obj_total               int
    num_max_clipping_planes     int
    obj_list                    list of strings
    LAC_list                    list of strings
    component_matrix            float ndarray( [num_obj_total, num_fields]
Methods
    phantom_def_matrix_decoder(self,M = None)
    load_phantom_definition_matrix(self,phantom_name = None)
    generate_volume_image(self, vol)
    generate_line_integral_single_projection_FPD(self,CBGeoFPD:geo.CBFPD)        
Notes
    
History: Created by ZYu prior to 20250614
         Revised by JZhang Prior to 20260404
============================================================================'''
class StaticLACPhantom():
    def __init__(self,phantom_name): 
        self.phantom_name = phantom_name
        
        # Phantom type: static LAC
        self.motion = False
        self.content = 'LAC'
        self.spectral = False
        
        # Phantom definition matrix
        M = self.load_phantom_definition_matrix(phantom_name)
        self.phantom_def_matrix = M

        # ---------------------------------------------------------------------
        # [oID LAC idxPre] [vx0] [ra,rb,rc] [vea] [veb] [vec] 
        # ---------------------------------------------------------------------
        nprim= 18 # predefined 
        self.num_obj_precedence = np.int32(np.sum(M[:,2])) # 2nd column: additive - 0 | precedence - 1
        self.num_obj_total = np.shape(M)[0] # returns num. of rows of M
        self.max_num_clip_planes = np.int32( (np.shape(M)[1] - nprim)/4 ) # [d, venormal]: <= reserved | > clipped
        
        # List of objects of paraboloid shapes
        self.object_list = self.phantom_def_matrix_decoder()
    
    def phantom_def_matrix_decoder(self,M = None):
        if M is None:
            M = self.phantom_def_matrix
        
        nobj = self.num_obj_total
        nclip_max= self.max_num_clip_planes
            
        # List of objects of paraboloid shapes
        obj_list = []
        # ---------------------------------------------------------------------
        # [oID LAC idxPre] [vx0] [ra,rb,rc] [vea] [veb] [vec] 
        # ---------------------------------------------------------------------
        for k in np.arange(nobj):
            # The array def of the kth object
            A = M[k,]
            shape = enm.QuadricShape(A[0])
            LAC = A[1]
            bprecedence = A[2]
            vcenter = A[3:6]
            vradii = A[6:9]
            ve0 = A[9:12]
            ve1 = A[12:15]
            ve2 = A[15:18]
            
            clip_matrix = np.reshape(A[18:],[nclip_max,4]) 
            # Number of effective clipping planes
            clip_matrix_value = ( np.sum(np.abs(clip_matrix),axis = 1) > EPS )
            nclip = np.sum(clip_matrix_value)  
            clip_matrix = clip_matrix[np.arange(nclip),]
            
            if np.linalg.norm(ve2) == 0:
                print(k)
            
            obj = QuadraticShape3D(
                shape,LAC, bprecedence, 
                vcenter, vradii,
                ve0, ve1, ve2,
                clip_matrix)
            
            obj_list.append(obj)
        
        return obj_list
        
    '''------------------------------------------------------------------------
    [F] - load_phantom_definition_matrix(self,phantom_name = None)
    [I] - phantom_name [optional]: default is None (string)
    [O] - phantom_def_matrix: 2D array defining the phantom 
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def load_phantom_definition_matrix(self,phantom_name = None):
        # Full path of the phantom lib
        if phantom_name is None:
            self.phantom_name = phantom_name
        
        # Import the corresponding module
        try:
            phan = importlib.import_module(phantom_name)
            phantom_def_matrix = phan.generate_phantom_definition_matrix()
            return phantom_def_matrix
        except ModuleNotFoundError:
            print('Invalid analytical phantom!')
    
    '''------------------------------------------------------------------------
    [F] - generate_volume_image_single_object(self,vol, shape)
    [I] - vol: Volume3D
    [I] - shape: QuadraticShape3D
    [O] - vol: updated Volume Image
    [H] - Created by ZYu prior to 20250614
        - Revised by JZhang prior to 20260404
        {Add torus equation}
    ------------------------------------------------------------------------'''
    def generate_volume_image_single_object(self, vol, shape):
        # Volume x,y,z coordinates
        dz = vol.dslice
        dy = vol.drow
        dx = vol.dcol
        nz = vol.nslice
        ny = vol.nrow
        nx = vol.ncol
        zoff = vol.oslice
        yoff = vol.orow
        xoff = vol.ocol
        nxy = nx*ny
        
        # x crd - horizontal | y crd vertical
        x = (np.arange(nx) - xoff)*dx
        y = (np.arange(ny) - yoff)*dy
        z = (np.arange(nz) - zoff)*dz
        xx = np.ones([ny,1])*x.reshape([1,nx])
        yy = y.reshape([ny,1])*np.ones([1,nx])
        
        # Paraboloid shape characteristics        
        shape_cha = shape.calculate_shape_characteristics()
        vx0 = shape_cha['center']
        M = shape_cha['transform_matrix']
        EQ = shape_cha['equation_value']
        
        # precalculations
        xxc = xx - vx0[0]
        yyc = yy - vx0[1]
        zc = z  - vx0[2]
        
        nclip = np.shape(shape.clip_matrix)[0]
        LAC = shape.LAC
        torus_r = shape.vradii[0]
        torus_R = shape.vradii[1]
        bprecedence = shape.bprecedence

        for kz in range(nz):
            # ---- Primary part
            if shape.shape.value == 6:
                # X: [3, nxy]
                X = np.vstack([xxc.reshape([1, nxy]), yyc.reshape([1, nxy]), zc[kz] * np.ones([1, nxy])])
                # M^T * X => [3, nxy]
                XM = M.transpose().dot(X)
                # XMX: [1,nxy]
                XMX = np.sum(np.multiply(XM, X), axis=0)
                XX = np.sum(np.multiply(X, X), axis=0)
                eq_torus = (XX + np.ones_like(XX) * (torus_R ** 2 - torus_r ** 2)) ** 2 - 4 * torus_R ** 2 * XMX
                ii = (eq_torus <= EQ).reshape([ny, nx])

            else:
                # X: [3, nxy]
                X = np.vstack([xxc.reshape([1, nxy]), yyc.reshape([1, nxy]),zc[kz]*np.ones([1, nxy])])
                # M^T * X => [3, nxy]
                XM = M.transpose().dot(X)
                # XMX: [1,nxy]
                XMX = np.multiply(XM, X)
                eq_ellipsoid = np.sum(XMX, axis = 0)
                ii = (eq_ellipsoid <= EQ).reshape([ny, nx])

            I = vol.volume[kz].reshape([ny,nx])
            # clipping planes
            for kclip in range(nclip):
                dist = shape.clip_matrix[kclip,0]
                vnormal = shape.clip_matrix[kclip,1:4]
                eq_plane = xxc*vnormal[0] + yyc*vnormal[1] + zc[kz]*vnormal[2]
                jj  = (eq_plane <= dist)
                ii  = (ii & jj)

            I[ii] = LAC + I[ii]*(not bprecedence)

            vol.volume[kz] = I.reshape([1,ny,nx])
            
        return vol
            
    '''------------------------------------------------------------------------
    [F] - generate_volume_image(self,voxel_size, num_voxel, voxel_offset,kev = 0):
    [I] - vol: image:Volume3D
    [O] - vol: updated volume 3D image
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def generate_volume_image(self, vol): 
        print('Start generating digital volume for ', self.phantom_name)
        # Calculate contribution of each object
        for kobj in range(self.num_obj_total):
            obj = self.object_list[kobj]
            vol = self.generate_volume_image_single_object(vol,obj)
            # print('Processed obj #', kobj, 'with', np.shape(obj.clip_matrix)[0], 'clipping planes')
        
        return vol
    

    '''------------------------------------------------------------------------
    [F] - generate_line_integral_single_projection
    [I] - CBGeoFPD: class of CB geometry for flat panel detector
    [O] - g: single projection of size [nv,nu] using precedence rule
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def generate_line_integral_single_projection(self,CBGeoFPD:geo.CBFPD):
        
        # If no precedence rule, g is initialized with 0
        g = self.calculate_line_segment_precedence(CBGeoFPD)

        g = self.calculate_line_segment_additive(CBGeoFPD,g)
        
        # output
        return g
    
    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_precedence(self,CBGeoFPD)
    [I] - CBGeoFPD: class of CB geometry for flat panel detector
    [O] - g: single projection of size [nv,nu] using precedence rule
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def calculate_line_segment_precedence(self,CBGeoFPD:geo.CBFPD):
        nprec = self.num_obj_precedence
        nv = CBGeoFPD.nrow
        nu = CBGeoFPD.nchannel
        
        # If there is no obj with precedence rule, initialize g and return
        if nprec == 0:
            g = np.zeros([nv,nu])
            return g
              
              
        # Initialize knot weight
        vknot = np.array([])
        nknot = 0
        
        # Initialize mu list
        vmu = np.array([0.0])
            
        # Initialize an empty 2D array
        nuv = nv*nu
        vts = np.empty((0,nuv))
        vte = np.empty((0,nuv))
        
    
        for kprec in np.arange(nprec):
            obj = self.object_list[kprec] # obj is QuadraticShape3D
            # print('Processing precedent obj #', kprec, ': ', obj.shape.name,'...')
            # ---- Case I: Convex Object
            if obj.shape.name.lower() in ['sphere','ellipsoid','cylinder']:  
                # Line intersections
                T = self.calculate_line_segment_single_object_convex(obj,CBGeoFPD)
                vts = np.vstack( ( vts, T[0,:].reshape([1,nuv]) ) )
                vte = np.vstack( ( vte, T[1,:].reshape([1,nuv]) ) )
                
                # LAC
                vmu = np.hstack( (vmu, obj.LAC) )
                
                # Knot weights
                vknot = np.hstack( (vknot, np.power(2,nknot)) )
                nknot = nknot + 1
                
             
            # ---- Case II: Nonconvex Object    
            else: # ['paraboloid','hyperboloid1','hyperboloid2', 'torus']
                
                # Line intersections
                T = self.calculate_line_segment_single_object_nonconvex(obj,CBGeoFPD)
                vts = np.vstack( (vts, T[0,:].reshape([1,nuv]) ) )
                vte = np.vstack( (vte, T[1,:].reshape([1,nuv]) ) )
                vts = np.vstack( (vts, T[2,:].reshape([1,nuv]) ) )
                vte = np.vstack( (vte, T[3,:].reshape([1,nuv]) ) ) 
                
                # LAC
                vmu = np.hstack( (vmu,obj.LAC,obj.LAC) )
                
                # Knot weights
                vknot = np.hstack( ( vknot, np.power(2,nknot), np.power(2,nknot+1) ) )
                nknot = nknot + 2
                
        
        # Sort the intersections t: vt - [2knot,nuv]        
        vt = np.vstack( (vts,vte) )
        sort_order = np.argsort(vt, axis = 0)
        vt = np.take_along_axis(vt, sort_order, axis = 0)
        
        # Rearange knot weight according to the t sort order
        ones_nuv = np.ones(nuv)
        vknot = np.hstack((vknot,-vknot)) # size - 2*knot
        vknot = np.outer(vknot,ones_nuv)  # size - [2*nknot,nuv]
        vknot = np.take_along_axis(vknot,sort_order,axis = 0)
        vknot_cumsum = np.cumsum(vknot,axis = 0)
        
        # iitest  = np.where(vknot_cumsum<0,True,False)
        # vknot_cumsum_test = vknot[iitest]
        # LAC indices: [2*nknot, nuv]
        idx_mu = np.int16(np.ceil(np.log2(vknot_cumsum + 0.8)))
        
        # Calculate line integral
        vmu = vmu[idx_mu] #[2*nknot,nuv]
        vt_diff = np.diff(vt,axis=0)
        
        g = np.sum(vt_diff*vmu[:-1,:],axis=0).reshape(nv,nu)

        return g  
        
    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_additive(self,CBGeoFPD)
    [I] - CBGeoFPD: class of CB geometry for flat panel detector
    [O] - g: single projection of size [nv,nu] using additive rule
    [H] - Created by ZYu prior to 20250614
        - Revised by JZhang prior to 20260404
        {Add line segmentation condition}
    ------------------------------------------------------------------------'''
    def calculate_line_segment_additive(self,CBGeoFPD:geo.CBFPD,g:np.array):
        nprec = self.num_obj_precedence
        nttl = self.num_obj_total
        nv = CBGeoFPD.nrow
        nu = CBGeoFPD.nchannel
        nuv = nv*nu
        SPD = CBGeoFPD.SPD
            
        # Initialize an empty 2D array
        vts = np.empty((0,nuv))
        vte = np.empty((0,nuv))      
    
        for kprec in np.arange(nprec,nttl):
            obj = self.object_list[kprec]
            # print('Processing additive obj #', kprec, ': ', obj.shape.name,'...')
            # ---- Case I: Convex Object
            if obj.shape.name.lower() in ['sphere','ellipsoid','cylinder']:  
                # Line intersections
                T = self.calculate_line_segment_single_object_convex(obj,CBGeoFPD)
                T0 = T.copy()
                T0[T0 >= SPD] = TMAX
                T = T0.reshape(T.shape)
                vts = T[0,:].reshape([1,nuv])
                vte = T[1,:].reshape([1,nuv])
                '''
                mask = (vts < TMAX) & (vte >= TMAX)
                if np.any(mask):
                    raise ValueError('The detector is too close to the source.')
                '''
                gk = (vte - vts) * obj.LAC

            # ---- Case II: Nonconvex Object    
            else: # ['paraboloid','hyperboloid1','hyperboloid2', 'torus']
                
                # Line intersections
                T = self.calculate_line_segment_single_object_nonconvex(obj,CBGeoFPD)
                # Line segmentation should be less than the distance from source to detector
                T0 = T.copy()
                T0[T0 >= SPD] = TMAX
                T = T0.reshape(T.shape)
                vts1 = T[0,:].reshape([1,nuv])
                vte1 = T[1,:].reshape([1,nuv])
                vts2 = T[2,:].reshape([1,nuv])
                vte2 = T[3,:].reshape([1,nuv])
                '''
                mask = (vts1 < TMAX) & (vte1 >= TMAX) | (vts2 < TMAX) & (vte2 >= TMAX)
                if np.any(mask):
                    raise ValueError('The detector is too close to the source.')
                '''
                gk = (vte2 + vte1 - vts2 - vts1) * obj.LAC
            
            # ---- Update the projection data
            g = g + gk.reshape(nv, nu)
        
        return g
        
    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_single_object_convex(obj,CBGeoFPD)
    [I] - obj: class of QuadraticShape3D
    [I] - CBGeometryFPD: class in module of geometry
    [O] - T: matrix of size [2,nu*nv] with the following content
            T[0,:]: ts 
            T[1,:]: te
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''  
    def calculate_line_segment_single_object_convex(self,obj:QuadraticShape3D,CBGeoFPD):
        # Calculate the starting and ending position of all rays in 1 view
        match obj.shape.value:
            case enm.QuadricShape.SPHERE.value:
                T = self.calculate_line_segment_sphere(obj,CBGeoFPD)
            case enm.QuadricShape.ELLIPSOID.value:
                T = self.calculate_line_segment_ellipsoid(obj,CBGeoFPD)
            case enm.QuadricShape.CYLINDER.value:
                T = self.calculate_line_segment_cylinder(obj,CBGeoFPD)
            case _:
                raise Exception('Input obj is not sphere, ellipsoid or cylinder!')
        
        # Plane clipping for convex        
        T = self.calculate_line_segment_plane(obj,CBGeoFPD,T[0,:],T[1,:])
        
        # fig, ax = plt.subplots()
        # gk = T[1,:] - T[0,:]
        # ax.imshow(gk.reshape(1064,1064),cmap='gray')
        # ax.set_title('gk')
        # plt.show()
        # # plt.pause(1)
        
        return T

    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_single_object_nonconvex(obj,CBGeoFPD)
    [I] - obj: class of QuadraticShape3D
    [I] - CBGeometryFPD: class in module of geometry
    [O] - T: matrix of size [4,nu*nv] with the following content
            T[0,:]: ts1 
            T[1,:]: te1
            T[2,:]: ts2 
            T[3,:]: te2
    [H] - Created by ZYu prior to 20250614
        - Revised by JZhang prior to 20260404
        {Add calculation function of torus-line segmentation}
    ------------------------------------------------------------------------'''  
    def calculate_line_segment_single_object_nonconvex(self,obj:QuadraticShape3D,CBGeoFPD:geo.CBFPD):        
        # Calculate the starting and ending position of all rays in 1 view
        if obj.shape.name.lower() in ['cone','hyperboloid1','hyperboloid2']:
            T = self.calculate_line_segment_paraboloid(obj,CBGeoFPD)    
        elif obj.shape.name.lower() in ['torus']:
            T = self.calculate_line_segment_torus(obj,CBGeoFPD)
        else:
            raise Exception('Input obj is not a paraboloid!')
        
        # Plane clipping for convex
        T1 = self.calculate_line_segment_plane(obj,CBGeoFPD,T[0,:],T[1,:])
        T2 = self.calculate_line_segment_plane(obj,CBGeoFPD,T[2,:],T[3,:])
        
        T = np.vstack([T1,T2])
        
        return T
        
    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_plane(self,obj,CBGeoFPD,vts = None, vte = None) 
    [I] - obj: class of QuadraticShape3D
                clip_matrix = obj.clip_matrix
                clip_matrix: [nclip,4]
                clip_matrix[0,:] = [d,ve] w/ d as scalar and ve as a vector
    [I] - CBGeometryFPD: class in module of geometry
    [O] - T: matrix of size [2,nu*nv] with the following content
            T[0,:]: ts1 
            T[1,:]: te1
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def calculate_line_segment_plane(self,obj:QuadraticShape3D,CBGeoFPD:geo.CBFPD,vts: np.array = None, vte: np.array = None):            
        # Object: center and clip matrix 
        shape_cha = obj.calculate_shape_characteristics()
        vx0 = shape_cha['center'] # center (vx0）
        MClip = obj.clip_matrix
        nclip = np.size(MClip,0)
        
        # CB geometry related information
        va = CBGeoFPD.vsource
        nu = CBGeoFPD.nchannel
        nv = CBGeoFPD.nrow
        nuv = nu*nv
        # a 3xnuv matirx of unit vectors
        valpha = CBGeoFPD.generate_divergent_cone_beam_unit_vector() 
        va0 = va - vx0 
        
        # Calculate vts and vte
        if vts is None:
            vts = np.zeros(nuv)
        if vte is None:
            vte = (TMAX+EPS)*np.ones(nuv) # EPS is added for ordering purposes when vts = TMAX and vte = TMAX+EPS
        
        for k in range(nclip):
            dk = MClip[k,0]
            vtheta = MClip[k,1:]                
            # (va - vx0)'*vtheta
            d_va0_theta = np.dot(va0,vtheta) # scalar
            # valpha'*vtheta
            d_alpha_theta = np.dot(vtheta,valpha) #[3,nuv]
            
            # Case I: va is outside
            # (va - vx0)'vtheta > dk & valpha'*vtheta < 0 (reserved portion)
            if d_va0_theta > dk:
                iineg = np.where(d_alpha_theta < 0, True, False)
                iipos = ~iineg
                tstar = (dk-d_va0_theta)/d_alpha_theta[iineg]
                vts[iineg] = np.maximum(vts[iineg],tstar) # va outside, pointing to the reserved portion
                vts[iipos] = TMAX # va outside, pointing to the clipped portion
            # Case II: va is inside
            else:
                iipos = np.where(d_alpha_theta > 0, True, False) 
                tstar = (dk - d_va0_theta)/d_alpha_theta[iipos]
                vte[iipos] = np.minimum(vte[iipos],tstar) # va inside, pointing to the clipped portion
            # case of d_alpha_theta <= 0 , no action needed: va inside, pointing to the reserved portion
        
        # vte = np.maximum(vts,vte)
        vts = np.where(vts>=vte,TMAX,vts)
        vte = np.where(vts>=vte,TMAX+EPS,vte)
        
        # Output
        T = np.zeros([2,nuv])
        T[0,:] = vts
        T[1,:] = vte    
        return T
        
    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_sphere(self,obj,CBGeoFPD,vts = None, vte = None) 
    [I] - obj: class of QuadraticShape3D
    [I] - CBGeometryFPD: class in module of geometry
    [O] - T: matrix of size [2,nu*nv] with the following content
            T[0,:]: ts1 
            T[1,:]: te1
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def calculate_line_segment_sphere(self,obj:QuadraticShape3D,CBGeoFPD:geo.CBFPD,vts: np.array = None, vte: np.array = None):
        # Object related information： (vx - vx0)'M(vx - vx0) = r
        shape_cha = obj.calculate_shape_characteristics()
        vx0 = shape_cha['center'] # center (vx0）
        # M = shape_cha['transform_matrix'] # transform_matrx (M)
        EQ = shape_cha['equation_value'] # equavation_value (r)
        
        # CB geometry related information
        va = CBGeoFPD.vsource
        nu = CBGeoFPD.nchannel
        nv = CBGeoFPD.nrow
        nuv = nu*nv
        # a 3xnuv matirx of unit vectors
        valpha = CBGeoFPD.generate_divergent_cone_beam_unit_vector() 
        
        # Characteristic equiation
        va0 = va - vx0 
        B = np.dot( va0, valpha ) # B: [1,nuv]
        C = np.power(np.linalg.norm(va0),2) - EQ # C: scalar
        DELTA = np.power(B,2) - C # Delta = B*B - C: [1,nuv]
        
        # Calculate vts and vte
        if vts is None:
            vts = np.zeros(nuv)
        if vte is None:
            vte = (TMAX+EPS)*np.ones(nuv)
        
        iipos = np.where(DELTA > 0,True,False)
        iineg = ~iipos
        sqrt_delta = np.sqrt(DELTA[iipos])        
        ta = -B[iipos] - sqrt_delta
        tb = -B[iipos] + sqrt_delta
        
        vts[iipos] = np.maximum(vts[iipos],ta)
        vte[iipos] = np.minimum(vte[iipos],tb)
        
        vts[iineg] = TMAX # case of no intersection, ts = TMAX, te = TMAX+EPS, EPS is added to te for ordering purposes
        vte = np.maximum(vts,vte)
        
        # Output
        T = np.zeros([2,nuv])
        T[0,:] = vts
        T[1,:] = vte    
        return T
    
    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_ellipsoid(self,obj,CBGeoFPD,vts = None, vte = None) 
    [I] - obj: class of QuadraticShape3D
    [I] - CBGeometryFPD: class in module of geometry
    [O] - T: matrix of size [2,nu*nv] with the following content
            T[0,:]: ts1 
            T[1,:]: te1
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def calculate_line_segment_ellipsoid(self,obj:QuadraticShape3D,CBGeoFPD: geo.CBFPD,vts: np.array = None, vte: np.array = None):        
        # Object related information： (vx - vx0)'M(vx - vx0) = EQ
        shape_cha = obj.calculate_shape_characteristics()
        vx0 = shape_cha['center'] # center (vx0）
        M = shape_cha['transform_matrix'] # transform_matrx (M)
        EQ = shape_cha['equation_value'] # equavation_value (1)
        
        # CB geometry related information
        va = CBGeoFPD.vsource
        nu = CBGeoFPD.nchannel
        nv = CBGeoFPD.nrow
        nuv = nu*nv
        # a 3xnuv matirx of unit vectors
        valpha = CBGeoFPD.generate_divergent_cone_beam_unit_vector() 
                
        # Characteristic equiation
        va0 = va - vx0 
        # M is positive definie and A is positive
        A = np.sum(np.dot(valpha.T,M).T*valpha,axis=0) # A: [1,nuv]
        M0 = np.dot(va0,M)
        B = np.dot(M0,valpha) # B: [1,nuv]
        C = np.dot(M0,va0)-EQ # C: scalar
        DELTA = np.power(B,2) - C*A # Delta = B*B - A*C: [1,nuv]
        
        # Calculate vts and vte
        if vts is None:
            vts = np.zeros(nuv)
        if vte is None:
            vte = (TMAX+EPS)*np.ones(nuv)
        
        iipos = np.where(DELTA > 0,True,False)
        iineg = ~iipos
        sqrt_delta = np.sqrt(DELTA[iipos])        
        ta = -(B[iipos] + sqrt_delta)/A[iipos]
        tb = -(B[iipos] - sqrt_delta)/A[iipos]
        
        vts[iipos] = np.maximum(vts[iipos],ta)
        vte[iipos] = np.minimum(vte[iipos],tb)
        
        vts[iineg] = TMAX # case of no intersection, ts = TMAX, te = TMAX+EPS, EPS is added to te for ordering purposes
        
        vte = np.maximum(vts,vte)
        
        # Output
        T = np.zeros([2,nuv])
        T[0,:] = vts
        T[1,:] = vte    
        
        return T
    
    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_cylinder(self,obj,CBGeoFPD) 
    [I] - obj: class of QuadraticShape3D
    [I] - CBGeometryFPD: class in module of geometry
    [O] - T: matrix of size [2,nu*nv] with the following content
            T[0,:]: ts1 
            T[1,:]: te1
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def calculate_line_segment_cylinder(self,obj:QuadraticShape3D,CBGeoFPD:geo.CBFPD,vts:np.array = None, vte:np.array = None):        
        # Object related information： (vx - vx0)'M(vx - vx0) = r
        shape_cha = obj.calculate_shape_characteristics()
        vx0 = shape_cha['center'] # center (vx0）
        M = shape_cha['transform_matrix'] # transform_matrx (M) - Cylinder is convex and M is positive semidefinite
        EQ = shape_cha['equation_value'] # equavation_value (r)
        
        # CB geometry related information
        va = CBGeoFPD.vsource
        nu = CBGeoFPD.nchannel
        nv = CBGeoFPD.nrow
        nuv = nu*nv
        # a 3xnuv matirx of unit vectors
        valpha = CBGeoFPD.generate_divergent_cone_beam_unit_vector() 
        
        # Characteristic equiation
        va0 = va - vx0 
        # For cylinder, M is semi-positive, A>=0
        A = np.sum(np.dot(valpha.T,M).T * valpha,axis=0) # A: [1,nuv]
        M0 = np.dot(va0,M)
        B = np.dot(M0,valpha) # B: [1,nuv]
        C = np.dot(M0,va0)-EQ # C: scalar
        DELTA = np.power(B,2) - C * A # Delta = B*B - A*C: [1,nuv]
        
        # Calculate vts and vte
        if vts is None:
            vts = np.zeros(nuv)
        if vte is None:
            vte = (TMAX + EPS) * np.ones(nuv)
        
        # Case 1:  A > 0 and Delta >= 0
        ii1 = np.where((A > np.finfo(float).eps) & (DELTA >= 0), True, False)
        sqrt_delta = np.sqrt(DELTA[ii1])        
        ta = -(B[ii1] + sqrt_delta) / A[ii1]
        tb = -(B[ii1] - sqrt_delta) / A[ii1]
        vts[ii1] = np.maximum(vts[ii1], ta)
        vte[ii1] = np.minimum(vte[ii1], tb)
        
        # Case 2: A > 0 and Delta < 0: No solution
        ii2 = np.where((A > np.finfo(float).eps) & (DELTA < 0), True, False)
        vts[ii2] = TMAX
        
        # Case 3: A = 0 and C > 0: Line // axis, outside cylinder, no solution
        ii3 = np.where((A < np.finfo(float).eps) & (C > 0), True, False) 
        vts[ii3] = TMAX
        
        # Case 4: A = 0 and C <= 0: Line // axis, on/inside cylidner
        # no action is needed 
        
        vte = np.maximum(vts,vte)
            
        # Output
        T = np.zeros([2,nuv])
        T[0, :] = vts
        T[1, :] = vte
        
        return T
    
    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_paraboloid(self,obj,CBGeoFPD) - All three EQ = 0, 1 and -1
    [I] - obj: class of QuadraticShape3D
    [I] - CBGeometryFPD: class in module of geometry
    [O] - T: matrix of size [2,nu*nv] with the following content
            T[0,:]: ts1 
            T[1,:]: te1
            T[2,:]: ts2
            T[3,:]: te2
    [N] - This function applies to all three shapes
            paraboloid
            hyperboloid with two sheets
            hyperboloid with one sheet 
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def calculate_line_segment_paraboloid(self,obj:QuadraticShape3D,CBGeoFPD:geo.CBFPD,
                                            vts1:np.array = None, vte1:np.array = None,
                                            vts2:np.array = None, vte2:np.array = None):        
        # Object related information： (vx - vx0)'M(vx - vx0) = r
        shape_cha = obj.calculate_shape_characteristics()
        vx0 = shape_cha['center'] # center (vx0）
        M = shape_cha['transform_matrix'] # transform_matrx (M)
        EQ = shape_cha['equation_value'] # equavation_value (r)
        
        # CB geometry related information
        va = CBGeoFPD.vsource
        nu = CBGeoFPD.nchannel
        nv = CBGeoFPD.nrow
        nuv = nu*nv
        # a 3xnuv matirx of unit vectors
        valpha = CBGeoFPD.generate_divergent_cone_beam_unit_vector() 
        
        # Characteristic equiation
        va0 = va - vx0 
        # For cylinder, M is semi-positive, A>=0
        A = np.sum(np.dot(valpha.T,M).T*valpha,axis=0) # A: [1,nuv]
        M0 = np.dot(va0,M)
        B = np.dot(M0,valpha) # B: [1,nuv]
        C = np.dot(M0,va0)-EQ # C: scalar
        DELTA = np.power(B,2) - C*A # Delta = B*B - A*C: [1,nuv]
        
        # Ip = (vxinf - vx0)'M(vxinf - vx0) = vxinf0' * M * vxinf0
        # vxinf0 = va0.reshape([3,1]) - np.finfo(float).max*valpha # [3,nuv]
        vxinf0 = va0.reshape([3,1]) - TMAX*valpha # [3,nuv]
        Ip = np.sum(np.dot(vxinf0.T,M).T*vxinf0,axis=0)
        
        # Calculate vts and vte
        if vts1 is None:
            vts1 = np.zeros(nuv)
        if vte1 is None:
            vte1 = (TMAX+EPS)*np.ones(nuv)
        if vts2 is None:
            vts2 = np.zeros(nuv)
        if vte2 is None:
            vte2 = (TMAX+EPS)*np.ones(nuv)
        
        # ---------------------------------------------------------------------
        # Case 1: Delta >= 0 and A = 0 - line parallel to asymptotic line
        # Case 1.1:  A = 0 and Ip > 0 - Line pointing from inside to outside
        ii11 = np.where((np.abs(A) < np.finfo(float).eps) & (Ip > 0))
        vte1[ii11] = np.minimum(vte1[ii11],-C/(2*B[ii11]))
        
        # Case 1.2:  A = 0 and Ip < 0 - Line pointing from outside to inside
        ii12 = np.where((np.abs(A) < np.finfo(float).eps) & (Ip < 0))
        vts1[ii12] = np.maximum(vts1[ii12],-C/(2*B[ii12]))
        
        # Case 1.3:  A = 0 and Ip = 0 - Line is on the surface
        #            Infinity many solutions, no action
        
        # ---------------------------------------------------------------------
        # Case 2: Delta >= 0 & A > 0
        ii2 = np.where((A >= np.finfo(float).eps) & (DELTA >=0))
        sqrt_delta2 = np.sqrt(DELTA[ii2])
        ta2 = -(B[ii2]+sqrt_delta2)/A[ii2]
        tb2 = -(B[ii2]-sqrt_delta2)/A[ii2]
        vts1[ii2] = np.maximum(vts1[ii2],ta2)
        vte1[ii2] = np.minimum(vte1[ii2],tb2)     
        vts2[ii2] = TMAX   # So the 2nd line segment is 0
        
        # ---------------------------------------------------------------------
        # Case 3: Delta >= 0 & A < 0
        ii3 = np.where((A <= (-EPS)) & (DELTA >=0))
        sqrt_delta3 = np.sqrt(DELTA[ii3])
        ta3 = -(B[ii3] - sqrt_delta3)/A[ii3]
        tb3 = -(B[ii3] + sqrt_delta3)/A[ii3]
        
        vte1[ii3] = np.minimum(vte1[ii3],ta3)
        vts2[ii3] = np.maximum(vts2[ii3],tb3)   
        
        # ---------------------------------------------------------------------
        # Case 4: Delta < 0 - no solutions
        ii4 = np.where(DELTA < 0)
        vts1[ii4] = TMAX
        vts2[ii4] = TMAX
        
        
        vte1 = np.maximum(vts1,vte1)
        vte2 = np.maximum(vts2,vte2)
            
        # Output
        T = np.zeros([4,nuv])
        T[0,:] = vts1
        T[1,:] = vte1
        T[2,:] = vts2
        T[3,:] = vte2
        
        return T


    '''------------------------------------------------------------------------
       [F] - calculate_line_segment_torus(self,obj,CBGeoFPD)
       [I] - obj: class of QuadraticShape3D
       [I] - CBGeometryFPD: class in module of geometry
       [O] - T: matrix of size [2,nu*nv] with the following content
               T[0,:]: ts1 
               T[1,:]: te1
               T[2,:]: ts2
               T[3,:]: te2
       [H] - Created by JZhang prior to 20260116
       ------------------------------------------------------------------------'''
    def calculate_line_segment_torus(self, obj: QuadraticShape3D, CBGeoFPD: geo.CBFPD,
                                          vts1: np.array = None, vte1: np.array = None,
                                          vts2: np.array = None, vte2: np.array = None):
        # Object related information： ((vx-vx0)(vx-vx0)+R^2-r^2)^2 = 4R^2(x-x0)M(x-x0)
        shape_cha = obj.calculate_shape_characteristics()
        vx0 = shape_cha['center']  # center (vx0）
        M = shape_cha['transform_matrix']  # transform_matrx (M)
        EQ = shape_cha['equation_value']  # equation_value (r)
        r = obj.vradii[0]
        R = obj.vradii[1]

        # CB geometry related information
        va = CBGeoFPD.vsource
        nu = CBGeoFPD.nchannel
        nv = CBGeoFPD.nrow
        nuv = nu * nv
        # a 3xnuv matirx of unit vectors
        valpha = CBGeoFPD.generate_divergent_cone_beam_unit_vector()
        t_temp = np.full((nuv, 4), TMAX+EPS, dtype=float)

        # characteristic equation
        # x^4 + E*x^2 + F*x + G = 0
        va0 = va[:, None] - vx0[:, None]
        m = np.einsum('ik, ik->k', va0, valpha)
        n = np.einsum('ik, ik->k', va0, va0) + R ** 2 - r ** 2
        Mvalpha = M @ valpha
        Mva0 = M @ va0
        g = np.einsum('ik, ik->k', valpha, Mvalpha)
        k = np.einsum('ik, ik->k', va0, Mva0)
        l = np.einsum('ik, ik->k', va0, Mvalpha)

        E = - 2.0 * (m ** 2.0) + 2.0 * n - 4.0 * (R ** 2.0) * g
        F = 8.0 * ((R ** 2.0) * m * g - (R ** 2.0) * l)
        G = (m ** 2.0 - n) ** 2.0 + 4.0 * (R ** 2.0) * (2.0 * m * l - k - (m ** 2.0) * g)
        H = - (E ** 2.0) / 12.0 - G
        L = - (E ** 3.0) / 108.0 - (F ** 2.0) / 8.0 + E * G / 3.0

        # ---------------------------------------------------------------------
        # Case 1: F = 0 -- x^4 + E*x^2 + G = 0
        # maskF0 = np.abs(F) <= EPS
        scale_F = np.abs(m * g) + np.abs(l) + 1.0
        maskF0 = np.abs(m * g - l) <= EPS * scale_F
        if np.any(maskF0):
            DELTA1 = E[maskF0] ** 2.0 - 4.0 * G[maskF0]
            # Case 1.1: F = 0 & DELTA1 >= 0
            mask1 = DELTA1 >= 0.0
            ii1 = np.where(maskF0)[0][mask1]
            dt1 = DELTA1[mask1]
            t_temp_square1 = (- E[ii1] + np.sqrt(dt1)) / 2.0
            t_temp_square2 = (- E[ii1] - np.sqrt(dt1)) / 2.0
            # Case 1.1.1: one non-negative solution
            mask11 = t_temp_square1 >= 0.0
            if np.any(mask11):
                ii11 = ii1[mask11]
                t_temp[ii11, 0] = np.sqrt(t_temp_square1[mask11])
                t_temp[ii11, 1] = - np.sqrt(t_temp_square1[mask11])
            # Case 1.1.2: another non-negative solution
            mask12 = t_temp_square2 >= 0.0
            if np.any(mask12):
                ii12 = ii1[mask12]
                t_temp[ii12, 2] = np.sqrt(t_temp_square2[mask12])
                t_temp[ii12, 3] = - np.sqrt(t_temp_square2[mask12])

        # ----------------------------------------------------------------------
        # Case 2: F != 0 -- (x^2+alpha)^2 - [(2*alpha-E)x^2 - F*x + (alpha^2-G)] = 0
        maskF1 = ~maskF0
        if np.any(maskF1):
            DELTA2_cubic = (L / 2.0) ** 2.0 + (H / 3.0) ** 3.0
            alpha_temp = np.full((nuv, 3), -np.inf, dtype=float)
            # 2.1: DELTA2_cubic > 0
            mask21 = maskF1 & (DELTA2_cubic > EPS)
            if np.any(mask21):
                sqrt_DELTA = np.sqrt(DELTA2_cubic[mask21])
                uu = -L[mask21] / 2.0 + sqrt_DELTA
                vv = -L[mask21] / 2.0 - sqrt_DELTA
                u = np.where(np.abs(uu) >= np.abs(vv), np.cbrt(uu), np.cbrt(vv))
                v = -(H[mask21] / 3.0) / u
                alpha_temp[mask21, 0] = u + v
            # 2.2: DELTA2_cubic = 0
            mask22 = maskF1 & (np.abs(DELTA2_cubic) <= EPS)
            if np.any(mask22):
                # 2.2.1: H == 0
                mask211 = mask22 & (np.abs(H) <= EPS)
                if np.any(mask211):
                    alpha_temp[mask211, 0] = 0.0
                # 2.2.2: H != 0
                mask222 = mask22 & (np.abs(H) > EPS)
                if np.any(mask222):
                    H1 = 2.0 * (-H[mask222] / 3.0) ** (3.0 / 2.0)
                    mask2221 = np.abs(L[mask222] - H1) <= EPS
                    if np.any(mask2221):
                        id2221 = np.where(mask222)[0][mask2221]
                        H2 = np.sqrt(- H[id2221] / 3.0)
                        alpha_temp[id2221, 0] = H2
                        alpha_temp[id2221, 1] = - 2.0 * H2
                    mask2222 = np.abs(L[mask222] + H1) <= EPS
                    if np.any(mask2222):
                        id2222 = np.where(mask222)[0][mask2222]
                        H3 = np.sqrt(-H[id2222] / 3.0)
                        alpha_temp[id2222, 0] = - H3
                        alpha_temp[id2222, 1] = 2.0 * H3
            # 2.3: DELTA2_cubic < 0
            mask23 = maskF1 & (DELTA2_cubic < -EPS)
            if np.any(mask23):
                theta = np.arccos((3.0 * L[mask23]) / (2.0 * H[mask23]) * np.sqrt(- 3.0 / H[mask23]))
                coef = 2.0 * np.sqrt(- H[mask23] / 3.0)
                alpha_temp[mask23, 0] = coef * np.cos(theta / 3.0)
                alpha_temp[mask23, 1] = coef * np.cos((theta + 2.0 * np.pi) / 3.0)
                alpha_temp[mask23, 2] = coef * np.cos((theta + 4.0 * np.pi) / 3.0)
            alpha_temp = alpha_temp + E[:, None] / 6.0
            alpha = np.max(alpha_temp, axis=1)

            # Continue: solve two quadratics if 2*alpha - E > 0
            dd = 2 * alpha - E
            scale_dd = np.abs(E) + 1.0
            EPS_dd = EPS * scale_dd
            dd = np.where(np.abs(dd) <= EPS_dd, 0.0, dd)
            maskq = maskF1 & (dd >= 0.0)
            if np.any(maskq):
                iiq = np.where(maskq)[0]
                sqrt_dd = np.sqrt(dd[iiq])
                unstable_term = np.sign(F[iiq]) * np.sqrt(np.maximum(alpha[iiq] ** 2.0 - G[iiq], 0.0))
                mask_pos = dd[iiq] > 0.0
                if np.any(mask_pos):
                    term1 = F[iiq][mask_pos] / (2.0 * np.sqrt(dd[iiq][mask_pos]))
                    choose2 = (np.abs(dd[iiq][mask_pos]) < 1e-3 * np.abs(F[iiq][mask_pos]))
                    unstable_term[mask_pos] = np.where(choose2, unstable_term[mask_pos], term1)
                DELTA3_1 = - 2.0 * alpha[iiq] - E[iiq] - 4.0 * unstable_term
                DELTA3_2 = - 2.0 * alpha[iiq] - E[iiq] + 4.0 * unstable_term
                # quadratic 1
                maskq1 = DELTA3_1 >= 0.0
                if np.any(maskq1):
                    iiq1 = iiq[maskq1]
                    dtq1 = DELTA3_1[maskq1]
                    t_temp[iiq1, 0] = (sqrt_dd[maskq1] - np.sqrt(dtq1)) / 2.0
                    t_temp[iiq1, 1] = (sqrt_dd[maskq1] + np.sqrt(dtq1)) / 2.0
                # quadratic 2
                maskq2 = DELTA3_2 >= 0.0
                if np.any(maskq2):
                    iiq2 = iiq[maskq2]
                    dtq2 = DELTA3_2[maskq2]
                    t_temp[iiq2, 2] = (- sqrt_dd[maskq2] - np.sqrt(dtq2)) / 2.0
                    t_temp[iiq2, 3] = (- sqrt_dd[maskq2] + np.sqrt(dtq2)) / 2.0

        t_temp = t_temp - m[:, None]
        t0 = t_temp.copy()
        t0[t0 <= 0] = TMAX+EPS
        t0 = np.reshape(t0, t_temp.shape)
        t = np.sort(t0, axis=1)

        # ---------------------------------------------------------------
        # Output : determine start points and end points
        # Output : determine start points and end points
        T = TMAX * np.ones([4, nuv])
        t_mid01 = (t[:, 0] + t[:, 1]) / 2
        X01 = va0 + t_mid01 * valpha
        XX01 = np.einsum('ik, ik->k', X01, X01)
        MX01 = M @ X01
        XMX01 = np.einsum('ik, ik->k', X01, MX01)
        eq_torus01 = (XX01 + (R ** 2 - r ** 2)) ** 2 - 4 * (R ** 2) * XMX01
        t_mid12 = (t[:, 1] + t[:, 2]) / 2
        X12 = va0 + t_mid12 * valpha
        XX12 = np.einsum('ik, ik->k', X12, X12)
        MX12 = M @ X12
        XMX12 = np.einsum('ik, ik->k', X12, MX12)
        eq_torus12 = (XX12 + (R ** 2 - r ** 2)) ** 2 - 4 * (R ** 2) * XMX12
        t_mid23 = (t[:, 2] + t[:, 3]) / 2
        X23 = va0 + t_mid23 * valpha
        XX23 = np.einsum('ik, ik->k', X23, X23)
        MX23 = M @ X23
        XMX23 = np.einsum('ik, ik->k', X23, MX23)
        eq_torus23 = (XX23 + (R ** 2 - r ** 2)) ** 2 - 4 * (R ** 2) * XMX23
        mask011 = (eq_torus01 < EQ) & (eq_torus12 >= EQ) & (eq_torus23 >= EQ)
        T[0, mask011] = t[mask011, 0]
        T[1, mask011] = t[mask011, 1]
        mask012 = (eq_torus01 >= EQ) & (eq_torus12 < EQ) & (eq_torus23 >= EQ)
        T[0, mask012] = t[mask012, 1]
        T[1, mask012] = t[mask012, 2]
        mask013 = (eq_torus01 >= EQ) & (eq_torus12 >= EQ) & (eq_torus23 < EQ)
        T[0, mask013] = t[mask013, 2]
        T[1, mask013] = t[mask013, 3]
        mask021 = (eq_torus01 < EQ) & (eq_torus12 < EQ) & (eq_torus23 >= EQ)
        T[0, mask021] = t[mask021, 0]
        T[1, mask021] = t[mask021, 1]
        T[2, mask021] = t[mask021, 1]
        T[3, mask021] = t[mask021, 2]
        mask022 = (eq_torus01 < EQ) & (eq_torus12 >= EQ) & (eq_torus23 < EQ)
        T[0, mask022] = t[mask022, 0]
        T[1, mask022] = t[mask022, 1]
        T[2, mask022] = t[mask022, 2]
        T[3, mask022] = t[mask022, 3]
        mask023 = (eq_torus01 >= EQ) & (eq_torus12 < EQ) & (eq_torus23 < EQ)
        T[0, mask023] = t[mask023, 1]
        T[1, mask023] = t[mask023, 2]
        T[2, mask023] = t[mask023, 2]
        T[3, mask023] = t[mask023, 3]

        return T


'''============================================================================
Class       
    StaticLACPhantom: 
        No motion
        LAC direclty specified
Parent
    None
Attributes
    phantom_type                static
    attenuation_type            LAC
    num_obj_precedence          int
    num_obj_total               int
    num_max_clipping_planes     int
    obj_list                    list of strings
    LAC_list                    list of strings
    component_matrix            float ndarray( [num_obj_total, num_fields]
Methods
    phantom_def_matrix_decoder(self,M = None)
    load_phantom_definition_matrix(self,phantom_name = None)
    generate_volume_image(self, vol)
    generate_line_integral_single_projection_FPD(self,CBGeoFPD:geo.CBFPD)        
Notes

History: Created by JZhang prior to 20260131
============================================================================'''


class StaticLACPhantom2D():
    def __init__(self, phantom_name):
        self.phantom_name = phantom_name

        # Phantom type: static LAC
        self.motion = False
        self.content = 'LAC'
        self.spectral = False

        # Phantom definition matrix
        M = self.load_phantom_definition_matrix(phantom_name)
        self.phantom_def_matrix = M

        # ---------------------------------------------------------------------
        # [oID LAC idxPre] [vx0] [ra,rb,rc] [vea] [veb] [vec]
        # ---------------------------------------------------------------------
        nprim = 18  # predefined
        self.num_obj_precedence = np.int32(np.sum(M[:, 2]))  # 2nd column: additive - 0 | precedence - 1
        self.num_obj_total = np.shape(M)[0]  # returns num. of rows of M
        self.max_num_clip_planes = np.int32((np.shape(M)[1] - nprim) / 4)  # [d, venormal]: <= reserved | > clipped

        # List of objects of paraboloid shapes
        self.object_list = self.phantom_def_matrix_decoder()

    def phantom_def_matrix_decoder(self, M=None):
        if M is None:
            M = self.phantom_def_matrix

        nobj = self.num_obj_total
        nclip_max = self.max_num_clip_planes

        # List of objects of paraboloid shapes
        obj_list = []
        # ---------------------------------------------------------------------
        # [oID LAC idxPre] [vx0] [ra,rb,rc] [vea] [veb] [vec]
        # ---------------------------------------------------------------------
        for k in np.arange(nobj):
            # The array def of the kth object
            A = M[k,]
            shape = enm.QuadricShape(A[0])
            LAC = A[1]
            bprecedence = A[2]
            vcenter = A[3:6]
            vradii = A[6:9]
            ve0 = A[9:12]
            ve1 = A[12:15]
            ve2 = A[15:18]

            clip_matrix = np.reshape(A[18:], [nclip_max, 4])
            # Number of effective clipping planes
            clip_matrix_value = (np.sum(np.abs(clip_matrix), axis=1) > EPS)
            nclip = np.sum(clip_matrix_value)
            clip_matrix = clip_matrix[np.arange(nclip),]

            if np.linalg.norm(ve2) == 0:
                print(k)

            obj = QuadraticShape3D(
                shape, LAC, bprecedence,
                vcenter, vradii,
                ve0, ve1, ve2,
                clip_matrix)

            obj_list.append(obj)

        return obj_list

    '''------------------------------------------------------------------------
    [F] - load_phantom_definition_matrix(self,phantom_name = None)
    [I] - phantom_name [optional]: default is None (string)
    [O] - phantom_def_matrix: 2D array defining the phantom 
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def load_phantom_definition_matrix(self, phantom_name=None):
        # Full path of the phantom lib
        if phantom_name is None:
            self.phantom_name = phantom_name

        # Import the corresponding module
        try:
            phan = importlib.import_module(phantom_name)
            phantom_def_matrix = phan.generate_phantom_definition_matrix()
            return phantom_def_matrix
        except ModuleNotFoundError:
            print('Invalid analytical phantom!')

    '''------------------------------------------------------------------------
    [F] - generate_volume_image_single_object(self,vol, shape)
    [I] - vol: Volume3D
    [I] - shape: QuadraticShape3D
    [O] - vol: updated Volume Image
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def generate_volume_image_single_object(self, vol, shape):
        # Volume x,y,z coordinates
        dz = vol.dslice
        dy = vol.drow
        dx = vol.dcol
        nz = vol.nslice
        ny = vol.nrow
        nx = vol.ncol
        zoff = vol.oslice
        yoff = vol.orow
        xoff = vol.ocol
        nxy = nx * ny

        # x crd - horizontal | y crd vertical
        x = (np.arange(nx) - xoff) * dx
        y = (np.arange(ny) - yoff) * dy
        z = (np.arange(nz) - zoff) * dz
        xx = np.ones([ny, 1]) * x.reshape([1, nx])
        yy = y.reshape([ny, 1]) * np.ones([1, nx])

        # Paraboloid shape characteristics
        shape_cha = shape.calculate_shape_characteristics()
        vx0 = shape_cha['center']
        M = shape_cha['transform_matrix']
        EQ = shape_cha['equation_value']

        # precalculations
        xxc = xx - vx0[0]
        yyc = yy - vx0[1]
        zc = z - vx0[2]

        nclip = np.shape(shape.clip_matrix)[0]
        LAC = shape.LAC
        torus_r = shape.vradii[0]
        torus_R = shape.vradii[1]
        bprecedence = shape.bprecedence

        for kz in range(nz):
            # ---- Primary part
            if shape.shape.value == 6:
                # X: [3, nxy]
                X = np.vstack([xxc.reshape([1, nxy]), yyc.reshape([1, nxy]), zc[kz] * np.ones([1, nxy])])
                # M^T * X => [3, nxy]
                XM = M.transpose().dot(X)
                # XMX: [1,nxy]
                XMX = np.sum(np.multiply(XM, X), axis=0)
                XX = np.sum(np.multiply(X, X), axis=0)
                # eq_torus = (XMX + np.ones_like(XMX)*(torus_R**2-torus_r**2))**2 - 4*torus_R**2*XM01X
                eq_torus = (XX + np.ones_like(XX) * (torus_R ** 2 - torus_r ** 2)) ** 2 - 4 * torus_R ** 2 * XMX
                ii = (eq_torus <= EQ).reshape([ny, nx])
            else:
                # X: [3, nxy]
                X = np.vstack([xxc.reshape([1, nxy]), yyc.reshape([1, nxy]), zc[kz] * np.ones([1, nxy])])
                # M^T * X => [3, nxy]
                XM = M.transpose().dot(X)
                # XMX: [1,nxy]
                XMX = np.multiply(XM, X)
                eq_ellipsoid = np.sum(XMX, axis=0)
                ii = (eq_ellipsoid <= EQ).reshape([ny, nx])

            I = vol.volume[kz].reshape([ny, nx])
            # clipping planes
            for kclip in range(nclip):
                dist = shape.clip_matrix[kclip, 0]
                vnormal = shape.clip_matrix[kclip, 1:4]
                eq_plane = xxc * vnormal[0] + yyc * vnormal[1] + zc[kz] * vnormal[2]
                jj = (eq_plane <= dist)
                ii = (ii & jj)

            I[ii] = LAC + I[ii] * (not bprecedence)

            vol.volume[kz] = I.reshape([1, ny, nx])

        return vol

    '''------------------------------------------------------------------------
    [F] - generate_volume_image(self,voxel_size, num_voxel, voxel_offset,kev = 0):
    [I] - vol: image:Volume3D
    [O] - vol: updated volume 3D image
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def generate_volume_image(self, vol):
        print('Start generating digital volume for ', self.phantom_name)
        # Calculate contribution of each object
        for kobj in range(self.num_obj_total):
            obj = self.object_list[kobj]
            vol = self.generate_volume_image_single_object(vol, obj)
            # print('Processed obj #', kobj, 'with', np.shape(obj.clip_matrix)[0], 'clipping planes')

        return vol

    '''------------------------------------------------------------------------
    [F] - generate_line_integral_single_projection_FPD
    [I] - CBGeoFPD: class of CB geometry for flat panel detector
    [O] - g: single projection of size [nv,nu] using precedence rule
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def generate_line_integral_single_projection_FPD_2D(self, CBGeoFPD: geo.CBFPD):

        # If no precedence rule, g is initialized with 0
        g = self.calculate_line_segment_precedence(CBGeoFPD)

        g = self.calculate_line_segment_additive(CBGeoFPD, g)

        # output
        return g

    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_precedence(self,CBGeoFPD)
    [I] - CBGeoFPD: class of CB geometry for flat panel detector
    [O] - g: single projection of size [nv,nu] using precedence rule
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def calculate_line_segment_precedence(self, CBGeoFPD: geo.CBFPD):
        nprec = self.num_obj_precedence
        nv = 1
        nu = CBGeoFPD.nchannel

        # If there is no obj with precedence rule, initialize g and return
        if nprec == 0:
            g = np.zeros([1, nu])
            return g

        # Initialize knot weight
        vknot = np.array([])
        nknot = 0

        # Initialize mu list
        vmu = np.array([0.0])

        # Initialize an empty 2D array
        nuv = nv * nu
        vts = np.empty((0, nuv))
        vte = np.empty((0, nuv))

        for kprec in np.arange(nprec):
            obj = self.object_list[kprec]  # obj is QuadraticShape3D
            # print('Processing precedent obj #', kprec, ': ', obj.shape.name,'...')
            # ---- Case I: Convex Object
            if obj.shape.name.lower() in ['sphere', 'ellipsoid', 'cylinder']:
                # Line intersections
                T = self.calculate_line_segment_single_object_convex(obj, CBGeoFPD)
                vts = np.vstack((vts, T[0, :].reshape([1, nuv])))
                vte = np.vstack((vte, T[1, :].reshape([1, nuv])))

                # LAC
                vmu = np.hstack((vmu, obj.LAC))

                # Knot weights
                vknot = np.hstack((vknot, np.power(2, nknot)))
                nknot = nknot + 1


            # ---- Case II: Nonconvex Object
            else:  # ['paraboloid','hyperboloid1','hyperboloid2']

                # Line intersections
                T = self.calculate_line_segment_single_object_nonconvex(obj, CBGeoFPD)
                vts = np.vstack((vts, T[0, :].reshape([1, nuv])))
                vte = np.vstack((vte, T[1, :].reshape([1, nuv])))
                vts = np.vstack((vts, T[2, :].reshape([1, nuv])))
                vte = np.vstack((vte, T[3, :].reshape([1, nuv])))

                # LAC
                vmu = np.hstack((vmu, obj.LAC, obj.LAC))

                # Knot weights
                vknot = np.hstack((vknot, np.power(2, nknot), np.power(2, nknot + 1)))
                nknot = nknot + 2

        # Sort the intersections t: vt - [2knot,nuv]
        vt = np.vstack((vts, vte))
        sort_order = np.argsort(vt, axis=0)
        vt = np.take_along_axis(vt, sort_order, axis=0)

        # Rearange knot weight according to the t sort order
        ones_nuv = np.ones(nuv)
        vknot = np.hstack((vknot, -vknot))  # size - 2*knot
        vknot = np.outer(vknot, ones_nuv)  # size - [2*nknot,nuv]
        vknot = np.take_along_axis(vknot, sort_order, axis=0)
        vknot_cumsum = np.cumsum(vknot, axis=0)

        # iitest  = np.where(vknot_cumsum<0,True,False)
        # vknot_cumsum_test = vknot[iitest]
        # LAC indices: [2*nknot, nuv]
        idx_mu = np.int16(np.ceil(np.log2(vknot_cumsum + 0.8)))

        # Calculate line integral
        vmu = vmu[idx_mu]  # [2*nknot,nuv]
        vt_diff = np.diff(vt, axis=0)

        g = np.sum(vt_diff * vmu[:-1, :], axis=0).reshape(nv, nu)

        return g

    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_additive(self,CBGeoFPD)
    [I] - CBGeoFPD: class of CB geometry for flat panel detector
    [O] - g: single projection of size [nv,nu] using additive rule
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def calculate_line_segment_additive(self, CBGeoFPD: geo.CBFPD, g: np.array):
        nprec = self.num_obj_precedence
        nttl = self.num_obj_total
        nv = 1
        nu = CBGeoFPD.nchannel
        nuv = nv * nu
        SPD = CBGeoFPD.SPD

        # Initialize an empty 2D array
        vts = np.empty((0, nuv))
        vte = np.empty((0, nuv))

        for kprec in np.arange(nprec, nttl):
            obj = self.object_list[kprec]
            # print('Processing additive obj #', kprec, ': ', obj.shape.name,'...')
            # ---- Case I: Convex Object
            if obj.shape.name.lower() in ['sphere', 'ellipsoid', 'cylinder']:
                # Line intersections
                T = self.calculate_line_segment_single_object_convex(obj, CBGeoFPD)
                T0 = T.copy()
                T0[T0 >= SPD] = TMAX
                T = T0.reshape(T.shape)
                vts = T[0, :].reshape([1, nuv])
                vte = T[1, :].reshape([1, nuv])

                gk = (vte - vts) * obj.LAC
                # ---- Case II: Nonconvex Object
            else:  # ['paraboloid','hyperboloid1','hyperboloid2', 'torus']

                # Line intersections
                T = self.calculate_line_segment_single_object_nonconvex(obj, CBGeoFPD)
                T0 = T.copy()
                T0[T0 >= SPD] = TMAX
                T = T0.reshape(T.shape)
                vts1 = T[0, :].reshape([1, nuv])
                vte1 = T[1, :].reshape([1, nuv])
                vts2 = T[2, :].reshape([1, nuv])
                vte2 = T[3, :].reshape([1, nuv])

                gk = (vte2 + vte1 - vts2 - vts1) * obj.LAC

            # ---- Update the projection data
            g = g + gk.reshape(nv, nu)

        return g

    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_single_object_convex(obj,CBGeoFPD)
    [I] - obj: class of QuadraticShape3D
    [I] - CBGeometryFPD: class in module of geometry
    [O] - T: matrix of size [2,nu*nv] with the following content
            T[0,:]: ts 
            T[1,:]: te
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def calculate_line_segment_single_object_convex(self, obj: QuadraticShape3D, CBGeoFPD):
        # Calculate the starting and ending position of all rays in 1 view
        match obj.shape.value:
            case enm.QuadricShape.SPHERE.value:
                T = self.calculate_line_segment_sphere(obj, CBGeoFPD)
            case enm.QuadricShape.ELLIPSOID.value:
                T = self.calculate_line_segment_ellipsoid(obj, CBGeoFPD)
            case enm.QuadricShape.CYLINDER.value:
                T = self.calculate_line_segment_cylinder(obj, CBGeoFPD)
            case _:
                raise Exception('Input obj is not sphere, ellipsoid or cylinder!')

        # Plane clipping for convex
        T = self.calculate_line_segment_plane(obj, CBGeoFPD, T[0, :], T[1, :])

        # fig, ax = plt.subplots()
        # gk = T[1,:] - T[0,:]
        # ax.imshow(gk.reshape(1064,1064),cmap='gray')
        # ax.set_title('gk')
        # plt.show()
        # # plt.pause(1)

        return T

    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_single_object_nonconvex(obj,CBGeoFPD)
    [I] - obj: class of QuadraticShape3D
    [I] - CBGeometryFPD: class in module of geometry
    [O] - T: matrix of size [4,nu*nv] with the following content
            T[0,:]: ts1 
            T[1,:]: te1
            T[2,:]: ts2 
            T[3,:]: te2
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def calculate_line_segment_single_object_nonconvex(self, obj: QuadraticShape3D, CBGeoFPD: geo.CBFPD):
        # Calculate the starting and ending position of all rays in 1 view
        if obj.shape.name.lower() in ['cone', 'hyperboloid1', 'hyperboloid2']:
            T = self.calculate_line_segment_paraboloid(obj, CBGeoFPD)
        elif obj.shape.name.lower() in ['torus']:
            T = self.calculate_line_segment_torus(obj, CBGeoFPD)
        else:
            raise Exception('Input obj is not a paraboloid!')

        # Plane clipping for convex
        T1 = self.calculate_line_segment_plane(obj, CBGeoFPD, T[0, :], T[1, :])
        T2 = self.calculate_line_segment_plane(obj, CBGeoFPD, T[2, :], T[3, :])

        T = np.vstack([T1, T2])

        return T

    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_plane(self,obj,CBGeoFPD,vts = None, vte = None) 
    [I] - obj: class of QuadraticShape3D
                clip_matrix = obj.clip_matrix
                clip_matrix: [nclip,4]
                clip_matrix[0,:] = [d,ve] w/ d as scalar and ve as a vector
    [I] - CBGeometryFPD: class in module of geometry
    [O] - T: matrix of size [2,nu*nv] with the following content
            T[0,:]: ts1 
            T[1,:]: te1
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def calculate_line_segment_plane(self, obj: QuadraticShape3D, CBGeoFPD: geo.CBFPD, vts: np.array = None,
                                     vte: np.array = None):
        # Object: center and clip matrix
        shape_cha = obj.calculate_shape_characteristics()
        vx0 = shape_cha['center']  # center (vx0）
        MClip = obj.clip_matrix
        nclip = np.size(MClip, 0)

        # CB geometry related information
        va = CBGeoFPD.vsource
        nu = CBGeoFPD.nchannel
        nv = 1
        nuv = nu * nv
        # a 3xnuv matirx of unit vectors
        valpha = CBGeoFPD.generate_divergent_cone_beam_unit_vector()
        va0 = va - vx0

        # Calculate vts and vte
        if vts is None:
            vts = np.zeros(nuv)
        if vte is None:
            vte = (TMAX + EPS) * np.ones(nuv)  # EPS is added for ordering purposes when vts = TMAX and vte = TMAX+EPS

        for k in range(nclip):
            dk = MClip[k, 0]
            vtheta = MClip[k, 1:]
            # (va - vx0)'*vtheta
            d_va0_theta = np.dot(va0, vtheta)  # scalar
            # valpha'*vtheta
            d_alpha_theta = np.dot(vtheta, valpha)  # [3,nuv]

            # Case I: va is outside
            # (va - vx0)'vtheta > dk & valpha'*vtheta < 0 (reserved portion)
            if d_va0_theta > dk:
                iineg = np.where(d_alpha_theta < 0, True, False)
                iipos = ~iineg
                tstar = (dk - d_va0_theta) / d_alpha_theta[iineg]
                vts[iineg] = np.maximum(vts[iineg], tstar)  # va outside, pointing to the reserved portion
                vts[iipos] = TMAX  # va outside, pointing to the clipped portion
            # Case II: va is inside
            else:
                iipos = np.where(d_alpha_theta > 0, True, False)
                tstar = (dk - d_va0_theta) / d_alpha_theta[iipos]
                vte[iipos] = np.minimum(vte[iipos], tstar)  # va inside, pointing to the clipped portion
            # case of d_alpha_theta <= 0 , no action needed: va inside, pointing to the reserved portion

        # vte = np.maximum(vts,vte)
        vts = np.where(vts >= vte, TMAX, vts)
        vte = np.where(vts >= vte, TMAX + EPS, vte)

        # Output
        T = np.zeros([2, nuv])
        T[0, :] = vts
        T[1, :] = vte
        return T

    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_sphere(self,obj,CBGeoFPD,vts = None, vte = None) 
    [I] - obj: class of QuadraticShape3D
    [I] - CBGeometryFPD: class in module of geometry
    [O] - T: matrix of size [2,nu*nv] with the following content
            T[0,:]: ts1 
            T[1,:]: te1
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def calculate_line_segment_sphere(self, obj: QuadraticShape3D, CBGeoFPD: geo.CBFPD, vts: np.array = None,
                                      vte: np.array = None):
        # Object related information： (vx - vx0)'M(vx - vx0) = r
        shape_cha = obj.calculate_shape_characteristics()
        vx0 = shape_cha['center']  # center (vx0）
        # M = shape_cha['transform_matrix'] # transform_matrx (M)
        EQ = shape_cha['equation_value']  # equavation_value (r)

        # CB geometry related information
        va = CBGeoFPD.vsource
        nu = CBGeoFPD.nchannel
        nv = CBGeoFPD.nrow
        nuv = nu * nv
        # a 3xnuv matirx of unit vectors
        valpha = CBGeoFPD.generate_divergent_cone_beam_unit_vector()

        # Characteristic equiation
        va0 = va - vx0
        B = np.dot(va0, valpha)  # B: [1,nuv]
        C = np.power(np.linalg.norm(va0), 2) - EQ  # C: scalar
        DELTA = np.power(B, 2) - C  # Delta = B*B - C: [1,nuv]

        # Calculate vts and vte
        if vts is None:
            vts = np.zeros(nuv)
        if vte is None:
            vte = (TMAX + EPS) * np.ones(nuv)

        iipos = np.where(DELTA > 0, True, False)
        iineg = ~iipos
        sqrt_delta = np.sqrt(DELTA[iipos])
        ta = -B[iipos] - sqrt_delta
        tb = -B[iipos] + sqrt_delta

        vts[iipos] = np.maximum(vts[iipos], ta)
        vte[iipos] = np.minimum(vte[iipos], tb)

        vts[iineg] = TMAX  # case of no intersection, ts = TMAX, te = TMAX+EPS, EPS is added to te for ordering purposes
        vte = np.maximum(vts, vte)

        # Output
        T = np.zeros([2, nuv])
        T[0, :] = vts
        T[1, :] = vte
        return T

    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_ellipsoid(self,obj,CBGeoFPD,vts = None, vte = None) 
    [I] - obj: class of QuadraticShape3D
    [I] - CBGeometryFPD: class in module of geometry
    [O] - T: matrix of size [2,nu*nv] with the following content
            T[0,:]: ts1 
            T[1,:]: te1
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def calculate_line_segment_ellipsoid(self, obj: QuadraticShape3D, CBGeoFPD: geo.CBFPD, vts: np.array = None,
                                         vte: np.array = None):
        # Object related information： (vx - vx0)'M(vx - vx0) = EQ
        shape_cha = obj.calculate_shape_characteristics()
        vx0 = shape_cha['center']  # center (vx0）
        M = shape_cha['transform_matrix']  # transform_matrx (M)
        EQ = shape_cha['equation_value']  # equavation_value (1)

        # CB geometry related information
        va = CBGeoFPD.vsource
        nu = CBGeoFPD.nchannel
        nv = CBGeoFPD.nrow
        nuv = nu * nv
        # a 3xnuv matirx of unit vectors
        valpha = CBGeoFPD.generate_divergent_cone_beam_unit_vector()

        # Characteristic equiation
        va0 = va - vx0
        # M is positive definie and A is positive
        A = np.sum(np.dot(valpha.T, M).T * valpha, axis=0)  # A: [1,nuv]
        M0 = np.dot(va0, M)
        B = np.dot(M0, valpha)  # B: [1,nuv]
        C = np.dot(M0, va0) - EQ  # C: scalar
        DELTA = np.power(B, 2) - C * A  # Delta = B*B - A*C: [1,nuv]

        # Calculate vts and vte
        if vts is None:
            vts = np.zeros(nuv)
        if vte is None:
            vte = (TMAX + EPS) * np.ones(nuv)

        iipos = np.where(DELTA > 0, True, False)
        iineg = ~iipos
        sqrt_delta = np.sqrt(DELTA[iipos])
        ta = -(B[iipos] + sqrt_delta) / A[iipos]
        tb = -(B[iipos] - sqrt_delta) / A[iipos]

        vts[iipos] = np.maximum(vts[iipos], ta)
        vte[iipos] = np.minimum(vte[iipos], tb)

        vts[iineg] = TMAX  # case of no intersection, ts = TMAX, te = TMAX+EPS, EPS is added to te for ordering purposes

        vte = np.maximum(vts, vte)

        # Output
        T = np.zeros([2, nuv])
        T[0, :] = vts
        T[1, :] = vte

        return T

    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_cylinder(self,obj,CBGeoFPD) 
    [I] - obj: class of QuadraticShape3D
    [I] - CBGeometryFPD: class in module of geometry
    [O] - T: matrix of size [2,nu*nv] with the following content
            T[0,:]: ts1 
            T[1,:]: te1
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def calculate_line_segment_cylinder(self, obj: QuadraticShape3D, CBGeoFPD: geo.CBFPD, vts: np.array = None,
                                        vte: np.array = None):
        # Object related information： (vx - vx0)'M(vx - vx0) = r
        shape_cha = obj.calculate_shape_characteristics()
        vx0 = shape_cha['center']  # center (vx0）
        M = shape_cha['transform_matrix']  # transform_matrx (M) - Cylinder is convex and M is positive semidefinite
        EQ = shape_cha['equation_value']  # equavation_value (r)

        # CB geometry related information
        va = CBGeoFPD.vsource
        nu = CBGeoFPD.nchannel
        nv = 1
        nuv = nu * nv
        # a 3xnuv matirx of unit vectors
        valpha = CBGeoFPD.generate_divergent_cone_beam_unit_vector()

        # Characteristic equiation
        va0 = va - vx0
        # For cylinder, M is semi-positive, A>=0
        A = np.sum(np.dot(valpha.T, M).T * valpha, axis=0)  # A: [1,nuv]
        M0 = np.dot(va0, M)
        B = np.dot(M0, valpha)  # B: [1,nuv]
        C = np.dot(M0, va0) - EQ  # C: scalar
        DELTA = np.power(B, 2) - C * A  # Delta = B*B - A*C: [1,nuv]

        # Calculate vts and vte
        if vts is None:
            vts = np.zeros(nuv)
        if vte is None:
            vte = (TMAX + EPS) * np.ones(nuv)

        # Case 1:  A > 0 and Delta >= 0
        ii1 = np.where((A > np.finfo(float).eps) & (DELTA >= 0), True, False)
        sqrt_delta = np.sqrt(DELTA[ii1])
        ta = -(B[ii1] + sqrt_delta) / A[ii1]
        tb = -(B[ii1] - sqrt_delta) / A[ii1]

        '''
        r_ta = A*ta**2 + 2*B*ta + C
        r_tb = A*tb**2 + 2*B*tb + C
        scale_ta = np.abs(A) * ta ** 2 + 2 * np.abs(B) * np.abs(ta) + np.abs(C) + 1.0
        scale_tb = np.abs(A) * tb ** 2 + 2 * np.abs(B) * np.abs(tb) + np.abs(C) + 1.0
        rel_ta = np.abs(r_ta) / scale_ta
        rel_tb = np.abs(r_tb) / scale_tb
        '''

        vts[ii1] = np.maximum(vts[ii1], ta)
        vte[ii1] = np.minimum(vte[ii1], tb)

        # Case 2: A > 0 and Delta < 0: No solution
        ii2 = np.where((A > np.finfo(float).eps) & (DELTA < 0), True, False)
        vts[ii2] = TMAX

        # Case 3: A = 0 and C > 0: Line // axis, outside cylinder, no solution
        ii3 = np.where((A < np.finfo(float).eps) & (C > 0), True, False)
        vts[ii3] = TMAX

        # Case 4: A = 0 and C <= 0: Line // axis, on/inside cylidner
        # no action is needed

        vte = np.maximum(vts, vte)

        # Output
        T = np.zeros([2, nuv])
        T[0, :] = vts
        T[1, :] = vte

        return T

    '''------------------------------------------------------------------------
    [F] - calculate_line_segment_paraboloid(self,obj,CBGeoFPD) - All three EQ = 0, 1 and -1
    [I] - obj: class of QuadraticShape3D
    [I] - CBGeometryFPD: class in module of geometry
    [O] - T: matrix of size [2,nu*nv] with the following content
            T[0,:]: ts1 
            T[1,:]: te1
            T[2,:]: ts2
            T[3,:]: te2
    [N] - This function applies to all three shapes
            paraboloid
            hyperboloid with two sheets
            hyperboloid with one sheet 
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def calculate_line_segment_paraboloid(self, obj: QuadraticShape3D, CBGeoFPD: geo.CBFPD,
                                          vts1: np.array = None, vte1: np.array = None,
                                          vts2: np.array = None, vte2: np.array = None):
        # Object related information： (vx - vx0)'M(vx - vx0) = r
        shape_cha = obj.calculate_shape_characteristics()
        vx0 = shape_cha['center']  # center (vx0）
        M = shape_cha['transform_matrix']  # transform_matrx (M)
        EQ = shape_cha['equation_value']  # equavation_value (r)

        # CB geometry related information
        va = CBGeoFPD.vsource
        nu = CBGeoFPD.nchannel
        nv = 1
        nuv = nu * nv
        # a 3xnuv matirx of unit vectors
        valpha = CBGeoFPD.generate_divergent_cone_beam_unit_vector()

        # Characteristic equiation
        va0 = va - vx0
        # For cylinder, M is semi-positive, A>=0
        A = np.sum(np.dot(valpha.T, M).T * valpha, axis=0)  # A: [1,nuv]
        M0 = np.dot(va0, M)
        B = np.dot(M0, valpha)  # B: [1,nuv]
        C = np.dot(M0, va0) - EQ  # C: scalar
        DELTA = np.power(B, 2) - C * A  # Delta = B*B - A*C: [1,nuv]

        # Ip = (vxinf - vx0)'M(vxinf - vx0) = vxinf0' * M * vxinf0
        # vxinf0 = va0.reshape([3,1]) - np.finfo(float).max*valpha # [3,nuv]
        vxinf0 = va0.reshape([3, 1]) - TMAX * valpha  # [3,nuv]
        Ip = np.sum(np.dot(vxinf0.T, M).T * vxinf0, axis=0)

        # Calculate vts and vte
        if vts1 is None:
            vts1 = np.zeros(nuv)
        if vte1 is None:
            vte1 = (TMAX + EPS) * np.ones(nuv)
        if vts2 is None:
            vts2 = np.zeros(nuv)
        if vte2 is None:
            vte2 = (TMAX + EPS) * np.ones(nuv)

        # ---------------------------------------------------------------------
        # Case 1: Delta >= 0 and A = 0 - line parallel to asymptotic line
        # Case 1.1:  A = 0 and Ip > 0 - Line pointing from inside to outside
        ii11 = np.where((np.abs(A) < np.finfo(float).eps) & (Ip > 0))
        vte1[ii11] = np.minimum(vte1[ii11], -C / (2 * B[ii11]))

        # Case 1.2:  A = 0 and Ip < 0 - Line pointing from outside to inside
        ii12 = np.where((np.abs(A) < np.finfo(float).eps) & (Ip < 0))
        vts1[ii12] = np.maximum(vts1[ii12], -C / (2 * B[ii12]))

        # Case 1.3:  A = 0 and Ip = 0 - Line is on the surface
        #            Infinity many solutions, no action

        # ---------------------------------------------------------------------
        # Case 2: Delta >= 0 & A > 0
        ii2 = np.where((A >= np.finfo(float).eps) & (DELTA >= 0))
        sqrt_delta2 = np.sqrt(DELTA[ii2])
        ta2 = -(B[ii2] + sqrt_delta2) / A[ii2]
        tb2 = -(B[ii2] - sqrt_delta2) / A[ii2]
        vts1[ii2] = np.maximum(vts1[ii2], ta2)
        vte1[ii2] = np.minimum(vte1[ii2], tb2)
        vts2[ii2] = TMAX  # So the 2nd line segment is 0

        # ---------------------------------------------------------------------
        # Case 3: Delta >= 0 & A < 0
        ii3 = np.where((A <= (-EPS)) & (DELTA >= 0))
        sqrt_delta3 = np.sqrt(DELTA[ii3])
        ta3 = -(B[ii3] - sqrt_delta3) / A[ii3]
        tb3 = -(B[ii3] + sqrt_delta3) / A[ii3]

        vte1[ii3] = np.minimum(vte1[ii3], ta3)
        vts2[ii3] = np.maximum(vts2[ii3], tb3)

        # ---------------------------------------------------------------------
        # Case 4: Delta < 0 - no solutions
        ii4 = np.where(DELTA < 0)
        vts1[ii4] = TMAX
        vts2[ii4] = TMAX

        vte1 = np.maximum(vts1, vte1)
        vte2 = np.maximum(vts2, vte2)

        # Output
        T = np.zeros([4, nuv])
        T[0, :] = vts1
        T[1, :] = vte1
        T[2, :] = vts2
        T[3, :] = vte2

        return T

    '''------------------------------------------------------------------------
       [F] - calculate_line_segment_paraboloid(self,obj,CBGeoFPD) - All three EQ = 0, 1 and -1
       [I] - obj: class of QuadraticShape3D
       [I] - CBGeometryFPD: class in module of geometry
       [O] - T: matrix of size [2,nu*nv] with the following content
               T[0,:]: ts1 
               T[1,:]: te1
               T[2,:]: ts2
               T[3,:]: te2
       [N] - This function applies to all three shapes
               paraboloid
               hyperboloid with two sheets
               hyperboloid with one sheet 
       [H] - Created by JZhang prior to 20250116
       ------------------------------------------------------------------------'''

    def calculate_line_segment_torus(self, obj: QuadraticShape3D, CBGeoFPD: geo.CBFPD,
                                     vts1: np.array = None, vte1: np.array = None,
                                     vts2: np.array = None, vte2: np.array = None):
        # Object related information： ((vx-vx0)(vx-vx0)+R^2-r^2)^2 = 4R^2(x-x0)M(x-x0)
        shape_cha = obj.calculate_shape_characteristics()
        vx0 = shape_cha['center']  # center (vx0）
        M = shape_cha['transform_matrix']  # transform_matrx (M)
        EQ = shape_cha['equation_value']  # equation_value (r)
        r = obj.vradii[0]
        R = obj.vradii[1]

        # CB geometry related information
        va = CBGeoFPD.vsource
        nu = CBGeoFPD.nchannel
        nv = 1
        nuv = nu * nv
        # a 3xnuv matirx of unit vectors
        valpha = CBGeoFPD.generate_divergent_cone_beam_unit_vector()
        t_temp = np.full((nuv, 4), TMAX + EPS, dtype=float)

        # characteristic equation
        # x^4 + E*x^2 + F*x + G = 0
        va0 = va[:, None] - vx0[:, None]
        m = np.einsum('ik, ik->k', va0, valpha)
        n = np.einsum('ik, ik->k', va0, va0) + R ** 2 - r ** 2
        Mvalpha = M @ valpha
        Mva0 = M @ va0
        g = np.einsum('ik, ik->k', valpha, Mvalpha)
        k = np.einsum('ik, ik->k', va0, Mva0)
        l = np.einsum('ik, ik->k', va0, Mvalpha)

        E = - 2.0 * (m ** 2.0) + 2.0 * n - 4.0 * (R ** 2.0) * g
        F = 8.0 * ((R ** 2.0) *  m * g - (R ** 2.0) * l)
        G = (m ** 2.0 - n) ** 2.0 + 4.0 * (R ** 2.0) * (2.0 * m * l - k - (m ** 2.0) * g)
        H = - (E ** 2.0) / 12.0 - G
        L = - (E ** 3.0) / 108.0 - (F ** 2.0) / 8.0 + E * G / 3.0

        # ---------------------------------------------------------------------
        # Case 1: F = 0 -- x^4 + E*x^2 + G = 0
        '''
        scaleF = (np.abs(E) + np.abs(G) + 1.0) ** (1.5)
        EPS_F = EPS * scaleF
        maskF0 = np.abs(F) <= EPS_F
        '''

        scale_F = np.abs(m * g) + np.abs(l) + 1.0
        maskF0 = np.abs(m*g - l) <= EPS * scale_F

        # maskF0 = F == 0
        if np.any(maskF0):
            DELTA1 = E[maskF0] ** 2.0 - 4.0 * G[maskF0]
            # Case 1.1: F = 0 & DELTA1 >= 0
            mask1 = DELTA1 >= 0.0
            ii1 = np.where(maskF0)[0][mask1]
            dt1 = DELTA1[mask1]
            t_temp_square1 = (- E[ii1] + np.sqrt(dt1)) / 2.0
            t_temp_square2 = (- E[ii1] - np.sqrt(dt1)) / 2.0
            # Case 1.1.1: one non-negative solution
            mask11 = t_temp_square1 >= 0.0
            if np.any(mask11):
                ii11 = ii1[mask11]
                t_temp[ii11, 0] = np.sqrt(t_temp_square1[mask11])
                t_temp[ii11, 1] = - np.sqrt(t_temp_square1[mask11])
            # Case 1.1.2: another non-negative solution
            mask12 = t_temp_square2 >= 0.0
            if np.any(mask12):
                ii12 = ii1[mask12]
                t_temp[ii12, 2] = np.sqrt(t_temp_square2[mask12])
                t_temp[ii12, 3] = - np.sqrt(t_temp_square2[mask12])

        # ----------------------------------------------------------------------
        # Case 2: F != 0 -- (x^2+alpha)^2 - [(2*alpha-E)x^2 - F*x + (alpha^2-G)] = 0
        maskF1 = ~maskF0
        if np.any(maskF1):
            DELTA2_cubic = (L / 2.0) ** 2.0 + (H / 3.0) ** 3.0
            alpha_temp = np.full((nuv, 3), -np.inf, dtype=float)
            # 2.1: DELTA2_cubic > 0
            mask21 = maskF1 & (DELTA2_cubic > EPS)
            if np.any(mask21):
                sqrt_DELTA = np.sqrt(DELTA2_cubic[mask21])
                uu = -L[mask21] / 2.0 + sqrt_DELTA
                vv = -L[mask21] / 2.0 - sqrt_DELTA
                u = np.where(np.abs(uu) >= np.abs(vv), np.cbrt(uu), np.cbrt(vv))
                v = -(H[mask21] / 3.0) / u
                alpha_temp[mask21, 0] = u + v

            # 2.2: DELTA2_cubic = 0
            mask22 = maskF1 & (np.abs(DELTA2_cubic) == EPS)
            if np.any(mask22):
                # 2.2.1: H == 0
                mask211 = mask22 & (np.abs(H) <= np.cbrt(EPS))
                if np.any(mask211):
                    alpha_temp[mask211, 0] = 0.0
                # 2.2.2: H != 0
                mask222 = mask22 & (np.abs(H) > np.cbrt(EPS))
                if np.any(mask222):
                    H1 = 2.0 * (-H[mask222] / 3.0) ** (3.0 / 2.0)
                    mask2221 = np.abs(L[mask222] - H1) <= EPS
                    if np.any(mask2221):
                        id2221 = np.where(mask222)[0][mask2221]
                        H2 = np.sqrt(- H[id2221] / 3.0)
                        alpha_temp[id2221, 0] = H2
                        alpha_temp[id2221, 1] = - 2.0 * H2
                    mask2222 = np.abs(L[mask222] + H1) <= EPS
                    if np.any(mask2222):
                        id2222 = np.where(mask222)[0][mask2222]
                        H3 = np.sqrt(-H[id2222] / 3.0)
                        alpha_temp[id2222, 0] = - H3
                        alpha_temp[id2222, 1] = 2.0 * H3

            # 2.3: DELTA2_cubic < 0
            mask23 = maskF1 & (DELTA2_cubic < -EPS)
            if np.any(mask23):
                theta = np.arccos((3.0 * L[mask23]) / (2.0 * H[mask23]) * np.sqrt(- 3.0 / H[mask23]))
                coef = 2.0 * np.sqrt(- H[mask23] / 3.0)
                alpha_temp[mask23, 0] = coef * np.cos(theta / 3.0)
                alpha_temp[mask23, 1] = coef * np.cos((theta + 2.0 * np.pi) / 3.0)
                alpha_temp[mask23, 2] = coef * np.cos((theta + 4.0 * np.pi) / 3.0)

            '''
            alpha_temp0 = np.zeros_like(alpha_temp, dtype=complex)
            for i in range(nuv):
                alpha_temp0[i] = np.roots([1, 0, H[i], L[i]])
            alpha_temp1 = np.where(np.abs(alpha_temp0.imag) < 1e-10, alpha_temp0.real, -np.inf)
            diff_alpha = np.sort(alpha_temp, axis=1) - np.sort(alpha_temp1, axis=1)
            
            res_cubic1 = alpha_temp**3 + alpha_temp*H[:, None] + L[:, None]
            res_cubic2 = alpha_temp1**3 + alpha_temp1*H[:, None] + L[:, None]
            scale_cubic = np.abs(alpha_temp)**3 + np.abs(H[:, None])*np.abs(alpha_temp) + np.abs(L[:, None])
            rel = np.abs(res_cubic1) / scale_cubic
            '''

            alpha_temp = alpha_temp + E[:, None] / 6.0
            alpha = np.max(alpha_temp, axis=1)

            # Continue: solve two quadratics if 2*alpha - E > 0
            dd = 2 * alpha - E
            scale_dd = np.abs(E) + 1.0
            EPS_dd = EPS * scale_dd
            dd = np.where(np.abs(dd) <= EPS_dd, 0.0, dd)
            maskq = maskF1 & (dd >= 0.0)
            if np.any(maskq):
                iiq = np.where(maskq)[0]
                sqrt_dd = np.sqrt(dd[iiq])
                # ELTA3_1 = - 2 * alpha[iiq] - E[iiq] - 2 * F[iiq] / sqrt_dd
                # DELTA3_2 = - 2 * alpha[iiq] - E[iiq] + 2 * F[iiq] / sqrt_dd
                unstable_term = np.sign(F[iiq]) * np.sqrt(np.maximum(alpha[iiq] ** 2.0 - G[iiq], 0.0))
                mask_pos = dd[iiq] > 0.0
                if np.any(mask_pos):
                    term1 = F[iiq][mask_pos] / (2.0 * np.sqrt(dd[iiq][mask_pos]))
                    choose2 = (np.abs(dd[iiq][mask_pos]) < 1e-3 * np.abs(F[iiq][mask_pos]))
                    unstable_term[mask_pos] = np.where(choose2, unstable_term[mask_pos], term1)
                DELTA3_1 = - 2.0 * alpha[iiq] - E[iiq] - 4.0 * unstable_term
                DELTA3_2 = - 2.0 * alpha[iiq] - E[iiq] + 4.0 * unstable_term

                # quadratic 1
                maskq1 = DELTA3_1 >= 0.0
                if np.any(maskq1):
                    iiq1 = iiq[maskq1]
                    dtq1 = DELTA3_1[maskq1]
                    t_temp[iiq1, 0] = (sqrt_dd[maskq1] - np.sqrt(dtq1)) / 2.0
                    t_temp[iiq1, 1] = (sqrt_dd[maskq1] + np.sqrt(dtq1)) / 2.0
                # quadratic 2
                maskq2 = DELTA3_2 >= 0.0
                if np.any(maskq2):
                    iiq2 = iiq[maskq2]
                    dtq2 = DELTA3_2[maskq2]
                    t_temp[iiq2, 2] = (- sqrt_dd[maskq2] - np.sqrt(dtq2)) / 2.0
                    t_temp[iiq2, 3] = (- sqrt_dd[maskq2] + np.sqrt(dtq2)) / 2.0


        t_temp0 = np.zeros_like(t_temp, dtype=complex)
        for i in range(nuv):
            t_temp0[i] = np.roots([1, 0, E[i], F[i], G[i]])
        t_temp = np.where(np.abs(t_temp0.imag) < 1e-10, t_temp0.real, 1e10)

        '''
        t_temp = np.sort(t_temp, axis=1)
        t_temp1 = np.sort(t_temp1, axis=1)
        diff_t = t_temp - t_temp1
        res_quartic = t_temp**4 + t_temp**2 * E[:, None] + t_temp * F[:, None] + G[:, None]
        res_quartic1 = t_temp1 ** 4 + t_temp1 ** 2 * E[:, None] + t_temp1 * F[:, None] + G[:, None]
        scale_quartic = np.abs(t_temp)**4 + np.abs(E[:, None]) * np.abs(t_temp)**2 + np.abs(F[:, None]) * np.abs(t_temp) + np.abs(G[:, None])
        rel_quartic = np.abs(res_quartic) / scale_quartic
        '''

        t_temp = t_temp - m[:, None]

        t0 = t_temp.copy()
        t0[t0 <= 0] = TMAX + EPS
        t0 = np.reshape(t0, t_temp.shape)
        t = np.sort(t0, axis=1)

        # ---------------------------------------------------------------
        # Output : determine start points and end points
        T = TMAX * np.ones([4, nuv])
        t_mid01 = (t[:, 0] + t[:, 1]) / 2
        X01 = va0 + t_mid01 * valpha
        XX01 = np.einsum('ik, ik->k', X01, X01)
        MX01 = M @ X01
        XMX01 = np.einsum('ik, ik->k', X01, MX01)
        eq_torus01 = (XX01 + (R ** 2 - r ** 2)) ** 2 - 4 * (R ** 2) * XMX01
        t_mid12 = (t[:, 1] + t[:, 2]) / 2
        X12 = va0 + t_mid12 * valpha
        XX12 = np.einsum('ik, ik->k', X12, X12)
        MX12 = M @ X12
        XMX12 = np.einsum('ik, ik->k', X12, MX12)
        eq_torus12 = (XX12 + (R ** 2 - r ** 2)) ** 2 - 4 * (R ** 2) * XMX12
        t_mid23 = (t[:, 2] + t[:, 3]) / 2
        X23 = va0 + t_mid23 * valpha
        XX23 = np.einsum('ik, ik->k', X23, X23)
        MX23 = M @ X23
        XMX23 = np.einsum('ik, ik->k', X23, MX23)
        eq_torus23 = (XX23 + (R ** 2 - r ** 2)) ** 2 - 4 * (R ** 2) * XMX23
        mask011 = (eq_torus01 < EQ) & (eq_torus12 >= EQ) & (eq_torus23 >= EQ)
        T[0, mask011] = t[mask011, 0]
        T[1, mask011] = t[mask011, 1]
        mask012 = (eq_torus01 >= EQ) & (eq_torus12 < EQ) & (eq_torus23 >= EQ)
        T[0, mask012] = t[mask012, 1]
        T[1, mask012] = t[mask012, 2]
        mask013 = (eq_torus01 >= EQ) & (eq_torus12 >= EQ) & (eq_torus23 < EQ)
        T[0, mask013] = t[mask013, 2]
        T[1, mask013] = t[mask013, 3]
        mask021 = (eq_torus01 < EQ) & (eq_torus12 < EQ) & (eq_torus23 >= EQ)
        T[0, mask021] = t[mask021, 0]
        T[1, mask021] = t[mask021, 1]
        T[2, mask021] = t[mask021, 1]
        T[3, mask021] = t[mask021, 2]
        mask022 = (eq_torus01 < EQ) & (eq_torus12 >= EQ) & (eq_torus23 < EQ)
        T[0, mask022] = t[mask022, 0]
        T[1, mask022] = t[mask022, 1]
        T[2, mask022] = t[mask022, 2]
        T[3, mask022] = t[mask022, 3]
        mask023 = (eq_torus01 >= EQ) & (eq_torus12 < EQ) & (eq_torus23 < EQ)
        T[0, mask023] = t[mask023, 1]
        T[1, mask023] = t[mask023, 2]
        T[2, mask023] = t[mask023, 2]
        T[3, mask023] = t[mask023, 3]

        return T

'''============================================================================
Class       
    StaticSpectralPhantom: 
        No motion
        Material Specified by XCOM
Parent
    None
Attributes
    phantom_type                static
    attenuation_type            material
    num_obj_precedence          int
    num_obj_total               int
    num_max_clipping_planes     int
    obj_list                    list of strings
    LAC_list                    list of strings
    phantom_def_matrix          float ndarray( [num_obj_total, num_fields]
    material_list               array of lists
Methods
    load_phantom_definition_matrix
    generate_line_integral_single_projection_FPD_PCD_InvariantFluence
    phantom_def_matrix_decoder(self,M = None)
    load_phantom_definition_matrix(self,phantom_name = None)
    generate_volume_image(self, vol)
    generate_line_integral_single_projection_FPD(self,CBGeoFPD:geo.CBFPD)      
Notes
    
History: 20250610 - Created by ZYu
============================================================================'''
class StaticSpectralPhantom(StaticLACPhantom):
    def __init__(self,phantom_name,energy_kev = 70): 
        self.phantom_name = phantom_name
        self.energy_kev = energy_kev
        # Phantom type: static LAC
        self.motion = False
        self.content = 'LAC'
        self.spectral = False
        
        # Phantom definition matrix
        M = self.load_phantom_definition_matrix(energy_kev,phantom_name)        

        # ---------------------------------------------------------------------
        # [oID LAC idxPre] [vx0] [ra,rb,rc] [vea] [veb] [vec] 
        # ---------------------------------------------------------------------
        nprim= 18 # predefined 
        self.num_obj_precedence = np.int32(np.sum(M[:,2])) # 2nd column: additive - 0 | precedence - 1
        self.num_obj_total = np.shape(M)[0] # returns num. of rows of M
        self.max_num_clip_planes = np.int32( (np.shape(M)[1] - nprim)/4 ) # [d, venormal]: <= reserved | > clipped
        
        # List of objects of paraboloid shapes
        self.object_list = self.phantom_def_matrix_decoder()
    
        
    '''------------------------------------------------------------------------
    [F] - load_phantom_definition_matrix(self,phantom_name = None)
    [I] - phantom_name [optional]: default is None (string)
    [O] - phantom_def_matrix: 2D array defining the phantom 
    [H] - 20250610: Created by ZYu
    ------------------------------------------------------------------------'''
    def load_phantom_definition_matrix(self,energy_kev,phantom_name = None):
        # Full path of the phantom lib
        if phantom_name is None:
            self.phantom_name = phantom_name
        
        # Import the corresponding module
        try:
            phan = importlib.import_module(phantom_name)
            phantom_def_matrix, material_list = phan.generate_phantom_definition_matrix(energy_kev)
            self.phantom_def_matrix = phantom_def_matrix
            self.material_list = material_list
            return phantom_def_matrix
        except ModuleNotFoundError:
            print('Invalid analytical phantom!')

    '''------------------------------------------------------------------------
    [F] - generate_line_integral_single_projection_FPD_PCD_InvariantFluence
    [I] - CBGeoFPD: class of CB geometry for flat panel detector
    [I] - phi: imaging chain response, an array, function of energy, uniformly distributed
    [I] - energy_kev: energy in keV, an array, energy grids, uniformly distributed
    [I] - bPoisson: boolean, if True, use Poisson model, default is False
    [O] - g: single projection of size [nv,nu]
    [N] - Detection mode: Photon counting 
          Invariant X-ray fluence across all detector pixels
    [H] - 20250610: Created by ZYu
    ------------------------------------------------------------------------'''
    def generate_line_integral_single_projection_FPD_PCD_InvariantFluence(self,CBGeoFPD:geo.CBFPD,phi:np.array,energy_kev: np.array,bPoisson:np.bool_ = False):
        
        nkev = np.size(energy_kev) # number of energy grids
        I0 = np.sum(phi)
        I1 = None
                
        for k in range(nkev):
            # Update phantom objects/LAC at each energy
            energyk_kev = energy_kev[k]
            M = self.load_phantom_definition_matrix(energyk_kev,self.phantom_name)
            self.object_list = self.phantom_def_matrix_decoder(M)

            # Calculate the line integral for each energy
            gk = self.generate_line_integral_single_projection_FPD(CBGeoFPD) # [nv, nu]
            I0k = phi[k] # [nv,nu]
            I1k = I0k*np.exp(-gk)
            
            if I1 is None:
                I1 = np.zeros_like(I1k) # Initialize I1 with zeros
            
            I1 += I1k
        
        # Logarithm operator
        if not bPoisson:
            g = -np.log(I1/I0) 
        # output
        return g