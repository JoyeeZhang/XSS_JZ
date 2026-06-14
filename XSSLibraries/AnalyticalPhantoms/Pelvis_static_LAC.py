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
def generate_phantom_definition_matrix(muwater = 1.0, mucortical = 2.5, muspongy = 1.5, dmuskin = 0.05,mumetal = 5.0,
                                       drskin = 3.0, atorso1 = 153.0, btorso1 = 103.0, ctorso = 500.0, 
                                       rfemur1 = 30.0, rprostate1 = 22.0,afiducial = 2.5, bfiducial = 0.5, cfiducial = 0.5):
    # Maximum number of clipping planes involved
    nclip_max = 2
    nclip = 4*nclip_max
    vclip0 = np.zeros([1,nclip])
    vclip0 = vclip0.flatten()
    vclip1 = copy.deepcopy(vclip0)
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
    
    # Torso dimensions and LCA: Object 1 & 2 (Cylinder)
    # drskin = 3
    # atorso1 = 153 # Cylinder
    # btorso1 = 103 # Cylinder
    # ctorso = 500
    atorso0 = atorso1 - drskin # Cylinder
    btorso0 = btorso1 - drskin # Cylinder
    # ctorso = 500
    vctorso = vcenter
    # mutorso0 = muwater
    mutorso1 = muwater - dmuskin 
    
    # Femur dimensions and LCA: Object 3,4,5,6 (sphere)    
    # rfemur1 = 30 # sphere
    rfemur0 = rfemur1-drskin # sphere
    vcfemurleft = [-0.5*atorso0,0,0]
    vcfemurright = [0.5*atorso0,0,0]
    # mufemur1 = mucortical
    # mufemur0 = muspongy
    
    # Prostate dimensions and LCA: Object 7,8 (sphere)   
    # rprostate1 = 25 # sphere
    rprostate0 = rprostate1 - drskin # sphere
    vcprostate = vcenter
    muprostate = 1.05*muwater
    # muprostate0 = muwater
    # muprostate1 = muwater - dmuskin
    
    # Fiducial dimensions and LCA: Object 9 (ellipsoid)
    # afiducial = 2.5 # ellipsoid
    # bfiducial = 0.5 # ellipsoid
    # cfiducial = 0.5 # ellipsoid
    vcfiducial = vcenter
    # mufiducial = mumetal
    
    # -------------------------------------------------------------------------
    # [ oID LAC idxPre [vx0] [ra,rb,rc] [vea] [veb] [vec] [d1 ve1] [d2 ve2] ]
    # -------------------------------------------------------------------------
    # Object 1: Torso skin - cylinder
    vprim = [2.0,   mutorso1, 0.0] + vctorso + [atorso1,btorso1,ctorso]
    vprim = np.hstack([vprim,vbasis])
    
    vclip1[0] = ctorso
    vclip1[1:4]  = vec
    vclip1[4] = ctorso
    vclip1[5:8] = -np.array(vec)
    vobj = np.hstack([vprim,vclip1])
    phantom_def_matrix = vobj
    
    # Object 2: Torso inside - cylinder
    vprim = [2,   dmuskin, 0] + vctorso + [atorso0,btorso0,ctorso]    
    vprim = np.hstack([vprim,vbasis])
    
    vclip1[0] = ctorso
    vclip1[1:4]  = vec
    vclip1[4] = ctorso
    vclip1[5:8] = -np.array(vec)
    vobj = np.hstack([vprim,vclip1])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    # Object 3: Left femur outside - sphere
    vprim = [0,  mucortical - muwater, 0] + vcfemurleft + [rfemur1,rfemur1,rfemur1]       # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])

    vobj = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    # Object 4: Left femur inside - sphere
    vprim = [0,  muspongy - mucortical, 0] + vcfemurleft + [rfemur0,rfemur0,rfemur0]       # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])

    vobj = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    # Object 5: Right femur outside - sphere
    vprim = [0,  mucortical - muwater, 0] + vcfemurright + [rfemur1,rfemur1,rfemur1]       # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])

    vobj = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    # Object 6: Right femur inside - sphere
    vprim = [0,  muspongy - mucortical, 0] + vcfemurright+ [rfemur0,rfemur0,rfemur0]       # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])

    vobj = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    # Object 7: Prostate inside - sphere
    vprim = [0,  muprostate - muwater, 0] + vcprostate + [rprostate0,rprostate0,rprostate0]    # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])

    vobj = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    # Object 8: Fiducial - ellipsoid
    vprim = [1,  mumetal - muprostate, 0] + vcfiducial + [afiducial,bfiducial,cfiducial]    # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])

    vobj = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    return phantom_def_matrix
    
    
 


