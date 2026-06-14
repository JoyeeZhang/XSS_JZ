import numpy as np

'''----------------------------------------------------------------------------
[F] generate_phantom_definition_matrix():   
[O] phantom_def_matrix: [nobj,ncol]
        each row defines 1 object as described below
        [ oID LAC idxPre [vx0] [ra,rb,rc] [vea] [veb] [vec] [d1 ve1] [d2 ve2] ]
        oID: quadratic shape ID
        LAC: pixel value
        idxPre: 1 - precedence (always at the beginning of the matrix)
                0 - addtive
        vx0: object center along the basis vectors defined below
        [ra,rb,rc]: half axes along the basis vectors
        [vea,veb,vev]: basis vectors
        [d1 ve1]: d1(scalar), ve1 (1x3) vector
                  let vx be a point at the volume
                  if vx'*ve1 <= d1
                      vx reserved
                  else
                      vx clipped
        There could be more clipping planes defined by additional [d,ve]
        All objects have to share the same maximum number of clipping planes
[N] Quadratic shape encoder/decoder
        Shape        oID
        shpere:       0
        ellipsoid:    1
        cylinder:     2
        paraboloid:   3
        hyperboloid1: 4 (hyperboloid with one sheet)
        hyperboloid2: 5 (hyperboloid with two sheets)
    Precedence rule
        Precedence is indicated by 1
        Always start with object with precedence rule
        The precedence objects are always listed with an ascending prirotiy
    Object with precedence rule
        Object 5 - Calotte (1.8)
        Object 6 - Brain matter (1.05)
        Object 7 - Frontal sinus (0)
    Unit: mm
----------------------------------------------------------------------------'''
def generate_phantom_definition_matrix(mugland = 1.15, mufat = 1.125, muskin = 1.15, rskin = 63, rfat = 60, rglandL = 15, rglandM = 10, rglandS = 5):
    # Maximum number of clipping planes involved
    nclip_max = 2
    nclip = 4*nclip_max
    vclip0 = np.zeros([1,nclip])
    vclip0 = vclip0.flatten()
    vea = [1.0,0.0,0.0]
    veb = [0.0,1.0,0.0]
    vec = [0.0,0.0,1.0]
    vbasis = vea+veb+vec
    phantom_def_matrix = []
    
    vcenter = [0.0,0.0,0.0]
    rcgland = 0.5*rfat
    
    # -------------------------------------------------------------------------
    # [ oID LAC idxPre [vx0] [ra,rb,rc] [vea] [veb] [vec] [d1 ve1] [d2 ve2] ]
    # -------------------------------------------------------------------------
    # Object 1: Skin - Sphere
    vprim = [0,   muskin, 0.0] + vcenter + [rskin,rskin,rskin]
    vprim = np.hstack([vprim,vbasis])
   
    vobj = np.hstack([vprim,vclip0])
    phantom_def_matrix = vobj
    
    # Object 2: Fat - Sphere
    vprim = [0,   mufat - muskin, 0.0] + vcenter + [rfat,rfat,rfat]
    vprim = np.hstack([vprim,vbasis])
   
    vobj = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    # Object 3: Large gland on the x axis
    vprim = [0,   mugland - mufat, 0.0] + [rcgland,0,0] + [rglandL,rglandL,rglandL]
    vprim = np.hstack([vprim,vbasis])
   
    vobj = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    # Object 4: Medium gland 120 degree from the x axis
    theta = 120.0*np.pi/180
    vprim = [0,   mugland - mufat, 0.0] + [rcgland*np.cos(theta),rcgland*np.sin(theta),0] + [rglandM,rglandM,rglandM]
    vprim = np.hstack([vprim,vbasis])
   
    vobj = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    # Object 5: Medium gland 240 degree from the x axis
    theta = 240.0*np.pi/180
    vprim = [0,   mugland - mufat, 0.0] + [rcgland*np.cos(theta),rcgland*np.sin(theta),0] + [rglandS,rglandS,rglandS]
    vprim = np.hstack([vprim,vbasis])
   
    vobj = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    
    return phantom_def_matrix
    
    
 


