import numpy as np

'''----------------------------------------------------------------------------
[F] generate_phantom_definition_matrix():   
[O] phantom_def_matrix: [nobj,ncol]
        each row defines 1 object as described below
        [ oID LAC idxPre [vx0] [ra,rb,rc] [vea] [veb] [vec] [d1 ve1] [d2 ve2] ]
        oID: quadratic shape ID
        LAC: pixel value
        idxPre: 1 - precedence (always at the beginning of the matrix)
                0 - additive
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
        The precedence objects are always listed with an ascending priority
    Object with precedence rule
        Object 5 - Calotte (1.8)
        Object 6 - Brain matter (1.05)
        Object 7 - Frontal sinus (0)
    Unit: mm
----------------------------------------------------------------------------'''


def generate_phantom_definition_matrix():
    # Maximum number of clipping planes involved
    nclip_max = 2
    nclip = 4 * nclip_max
    vclip0 = np.zeros([1, nclip])
    vclip0 = vclip0.flatten()
    vbasis = np.reshape(np.eye(3), [1, 9])
    vbasis = vbasis.flatten()
    phantom_def_matrix = []
    material_def_list = []

    # -------------------------------------------------------------------------
    # Additive objects
    # ---- Object 1:
    lac = 0.5
    vprim = [0, lac, 0,  # sphere, LAC, additive index
             0, 0, 0,  # center of the object
             200, 200, 200]  # [ra,rb,rc]
    vprim = np.hstack([vprim, vbasis])
    v1 = np.hstack([vprim, vclip0])
    phantom_def_matrix = v1

    vprim = [0, lac, 0,  # sphere, LAC, additive index
             0, 0, 0,  # center of the object
             200, 200, 200]  # [ra,rb,rc]
    vprim = np.hstack([vprim, vbasis])
    v2 = np.hstack([vprim, vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix, v2])

    return phantom_def_matrix





