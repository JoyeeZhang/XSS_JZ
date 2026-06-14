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
    vea0 = [1.0,0.0,0.0]
    veb0 = [0.0,1.0,0.0]
    vec0 = [0.0,0.0,1.0]
    vbasis0 = vea0+veb0+vec0
    phantom_def_matrix = []
    
    # Radii definition
    rbreast = 60.0 # This is median size of the breast
    dskin = 2.0
    robject_base_xy = 35.0
    robject_base_yz = 40.0
    rtumor = [5.0,5.0,5.0]
    rfibroglandular_sphere = [15,15,15]
    rfibroglandular_ellipsoid = [5.0, 2.5, 2.5] # typical size of fibroglandular tissue
    rfibroglandular_ellipsoid_base = 10
    rmicrocalcification = [0.5,1.0,1.5,2.0,2.5] # typical size (diameter) is 1 mm - 5 mm
    rfibroglandular_group = 10
    
    # Basis centers
    vcenter_breast = [0.0,0.0,0.0]
    vcenter_Fibro_group_xy = [0.0, robject_base_xy,0.0]
    vcenter_Fibro_group_yz = [0.0, 0.0, robject_base_yz]
    
    # LAC
    mu_breast = 0.95
    mu_skin = mu_breast + 0.05
    mu_fibroglandular = 1.03
    mu_microcalcification = 1.6
    mu_tumor = mu_fibroglandular + 0.015
       
    # -------------------------------------------------------------------------
    # [ oID LAC idxPre [vx0] [ra,rb,rc] [vea] [veb] [vec] [d1 ve1] [d2 ve2] ]
    # -------------------------------------------------------------------------
    # Base objects:  Breast + skin
    # -------------------------------------------------------------------------
    # Object 0: Breast + skin - Sphere
    vprim = [0,   mu_skin, 0.0] + vcenter_breast + [rbreast + dskin, rbreast + dskin, rbreast + dskin]
    vprim = np.hstack([vprim,vbasis0])
    
    vclip = [10, 0.0, 0.0, -1.0,
             0.0, 0.0, 0.0, 0.0]
   
    vobj = np.hstack([vprim,vclip])
    phantom_def_matrix = vobj
    
    # Object 1: Breast - Sphere
    vprim = [0,   mu_breast - mu_skin, 0.0] + vcenter_breast + [rbreast, rbreast, rbreast]
    vprim = np.hstack([vprim,vbasis0])
   
    vclip = [10, 0.0, 0.0, -1.0,
             0.0, 0.0, 0.0, 0.0]
    vobj = np.hstack([vprim,vclip])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    # -------------------------------------------------------------------------
    # Object group in XY plane: 2.x: Fibroglandular
    # -------------------------------------------------------------------------
    # Object 2.x: Fibroglandular group - Ellipsoid
    vcenter_Fibro_group = np.array(vcenter_Fibro_group_xy)
    for k in range(0,8):
        theta = k*np.pi/4
        vea = [np.cos(theta), np.sin(theta), 0.0]
        veb = [-np.sin(theta) , np.cos(theta), 0.0]
        vec = [0.0, 0.0, 1.0]
        vbasis = vea + veb + vec
        vcenter_object = vcenter_Fibro_group + np.array([rfibroglandular_ellipsoid_base*np.cos(theta), rfibroglandular_ellipsoid_base*np.sin(theta), 0.0])
        vcenter_object = vcenter_object.tolist()
        vprim = [1,   mu_fibroglandular - mu_breast, 0.0] + vcenter_object + rfibroglandular_ellipsoid
        vprim = np.hstack([vprim,vbasis])
    
        vobj = np.hstack([vprim,vclip0])
        phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    # -------------------------------------------------------------------------
    # Object group in XY plane: 3.x: Microcalcification
    # -------------------------------------------------------------------------
    # Object 3.x: Microcalcification group - Sphere
    dmicro = 5.0 # distance between microcalcifications
    for k in range(1,6):
        rmicro = rmicrocalcification[k-1]
        xk = -robject_base_xy - (k-3.0)*dmicro
        vcenter_object = [xk,0.0,0.0]
        vprim = [0,   mu_microcalcification - mu_breast, 0.0] + vcenter_object + [rmicro, rmicro, rmicro]
        vprim = np.hstack([vprim,vbasis])
    
        vobj = np.hstack([vprim,vclip0])
        phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
        
    # -------------------------------------------------------------------------
    # Object group in XY plane: 4.x: Tumor
    # -------------------------------------------------------------------------
    # Object 4.x: Spherical fibroglandular tissue - Sphere
    theta = -np.pi/4.0
    vcenter = [robject_base_xy*np.cos(theta), robject_base_xy*np.sin(theta), 0.0]
    vprim = [0,   mu_fibroglandular - mu_breast, 0.0] + vcenter + rfibroglandular_sphere
    vprim = np.hstack([vprim, vbasis])
    
    vclip = [10, 0.0, 0.0, 1.0,
             0.0, 0.0, 0.0, 0.0]
    vobj = np.hstack([vprim,vclip])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    # Object 4.x: Tumor - Sphere
    vprim = [0,   mu_tumor - mu_fibroglandular, 0.0] + vcenter + rtumor
    vprim = np.hstack([vprim, vbasis])
    
    vclip = [10, 0.0, 0.0, -1.0,
             0.0, 0.0, 0.0, 0.0]
    vobj = np.hstack([vprim,vclip])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    # -------------------------------------------------------------------------
    # Object group in YZ plane: 2.x: Fibroglandular
    # -------------------------------------------------------------------------
    # Object 2.x: Fibroglandular group - Ellipsoid
    vcenter_Fibro_group = np.array(vcenter_Fibro_group_yz)
    for k in range(0,8):
        theta = k*np.pi/4
        vea = [0.0, np.sin(theta), np.cos(theta)]
        veb = [0.0, np.cos(theta), -np.sin(theta)]
        vec = [1.0, 0.0, 0.0]
        vbasis = vea + veb + vec
        vcenter_object = vcenter_Fibro_group + np.array([0.0, rfibroglandular_ellipsoid_base*np.sin(theta), rfibroglandular_ellipsoid_base*np.cos(theta)])
        vcenter_object = vcenter_object.tolist()
        vprim = [1,   mu_fibroglandular - mu_breast, 0.0] + vcenter_object + rfibroglandular_ellipsoid
        vprim = np.hstack([vprim,vbasis])
    
        vobj = np.hstack([vprim,vclip0])
        phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
        
    # -------------------------------------------------------------------------
    # Object group in YZ plane: 3.x: Microcalcification
    # -------------------------------------------------------------------------
    # Object 3.x: Microcalcification group - Sphere
    dmicro = 5.0 # distance between microcalcifications
    theta = np.pi/6.0
    for k in range(1,6):
        rmicro = rmicrocalcification[k-1]
        yk = robject_base_yz*np.cos(theta) + (k-3.0)*dmicro
        zk = robject_base_yz*np.sin(theta)
        vcenter_object = [0.0,yk,zk]
        vprim = [0,   mu_microcalcification - mu_breast, 0.0] + vcenter_object + [rmicro, rmicro, rmicro]
        vprim = np.hstack([vprim,vbasis])
    
        vobj = np.hstack([vprim,vclip0])
        phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
        
    # -------------------------------------------------------------------------
    # Object group in YZ plane: 4.x: Tumor
    # -------------------------------------------------------------------------
    # Object 4.x: Spherical fibroglandular tissue - Sphere
    theta =  np.pi/6.0
    vcenter = [0.0,-robject_base_yz*np.cos(theta), robject_base_yz*np.sin(theta)]
    vprim = [0,   mu_fibroglandular - mu_breast, 0.0] + vcenter + rfibroglandular_sphere
    vprim = np.hstack([vprim, vbasis])
    
    vobj = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    # Object 4.x: Tumor - Sphere
    vprim = [0,   mu_tumor - mu_fibroglandular, 0.0] + vcenter + rtumor
    vprim = np.hstack([vprim, vbasis])
    vobj = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    
    return phantom_def_matrix
    
    
 


