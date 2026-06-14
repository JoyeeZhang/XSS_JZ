import numpy as np
import copy

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
def generate_phantom_definition_matrix(muprostate = 1.05, dmuskin = 0.05, mumetal = 5.0, drskin = 2.0, rprostate1 = 25.0, afiducial = 2.5, bfiducial = 0.5, cfiducial = 0.5):
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
    
    # LCA values
    # muwater = 1.0
    # mucortical = 2.8
    # muspongy = 1.8 
    # dmuskin = 0.05
    # mumetal = 10
    
    vcenter = [0.0,0.0,0.0]
    
    rprostate0 = rprostate1 - drskin # sphere
    vcprostate = vcenter

    vcfiducialA = [0, rprostate1*0.5, 0]
    vcfiducialB = [0, -0.5*rprostate1,0]
       
    # -------------------------------------------------------------------------
    # [ oID LAC idxPre [vx0] [ra,rb,rc] [vea] [veb] [vec] [d1 ve1] [d2 ve2] ]
    # -------------------------------------------------------------------------
    # Object 1: Prostate with skin
    vprim = [0,   muprostate-dmuskin, 0] + vcenter + [rprostate1,rprostate1,rprostate1]
    vprim = np.hstack([vprim,vbasis])
    
    vobj = np.hstack([vprim,vclip0])
    phantom_def_matrix = vobj
    
    # Object 2: Prostate without skin
    vprim = [0,   dmuskin, 0] + vcenter + [rprostate0,rprostate0,rprostate0]
    vprim = np.hstack([vprim,vbasis])
    
    vobj = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    # Object 3: Fiducial - ellipsoid
    vprim = [1,  mumetal - muprostate, 0] + vcfiducialA + [afiducial,bfiducial,cfiducial]    # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])

    vobj = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    # Object 4: Fiducial - ellipsoid
    vprim = [1,  mumetal - muprostate, 0] + vcfiducialB + [bfiducial,cfiducial,afiducial]    # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])

    vobj = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    return phantom_def_matrix
    
    
 


