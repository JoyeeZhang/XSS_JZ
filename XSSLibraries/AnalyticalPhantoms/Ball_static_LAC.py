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
def generate_phantom_definition_matrix():
    # Maximum number of clipping planes involved
    nclip_max = 2
    nclip = 4*nclip_max
    vclip0 = np.zeros([1,nclip])
    vclip0 = vclip0.flatten()
    vbasis = np.reshape(np.eye(3),[1,9])
    vbasis = vbasis.flatten()
    phantom_def_matrix = []
    
    # -------------------------------------------------------------------------
    # [ oID LAC idxPre [vx0] [ra,rb,rc] [vea] [veb] [vec] [d1 ve1] [d2 ve2] ]
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Additive objects
    # ---- Object 1: Sphere (1.0)
    vprim = [0,   1,   1,    # sphere, LAC, precedence index
             0,   0,   0,        # center of the object
             120, 120,120]          # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])
    v1 = np.hstack([vprim,vclip0])
    phantom_def_matrix = v1
    phantom_def_matrix = phantom_def_matrix.reshape(1,-1) # make it 2D, this is important if the phantom has one object only
    
    # ---- Object 2: Ellipsoid (0.8)
    vprim = [1,   0.8,   1,    # ellipsoid, LAC, precedence index
             0,   0,   0,        # center of the object
             100, 100,100]          # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])
    v2 = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,v2])
    
    # ---- Object 3: cylinder (1.2)
    vprim = [2,   0.1,   0,         # cylinder, LAC, precedence index
             0,   0,     0,         # center of the object
             50, 40,    30]         # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])
    vbasis1 = [30, 0,0, 1]
    vbasis2 = [30, 0,0,-1]
    
    v3 = np.hstack([vprim,vbasis1,vbasis2])
    # phantom_def_matrix = np.vstack([phantom_def_matrix,v3])
    
     # ---- Object 4: cone (1.2)
    vprim = [3,   0.1,   0,         # cylinder, LAC, precedence index
             0,   0,     0,         # center of the object
             15, 20,    25]         # [ra,rb,rc]
    vbasis4 = [0,0,1,0,1,0,1,0,0]
    vprim = np.hstack([vprim,vbasis4])
    vbasis1 = [25, 1,0, 0]
    vbasis2 = [25,-1,0, 0]
    
    v3 = np.hstack([vprim,vbasis1,vbasis2])
    # phantom_def_matrix = np.vstack([phantom_def_matrix,v3])
   
    
    return phantom_def_matrix
    
    
 


