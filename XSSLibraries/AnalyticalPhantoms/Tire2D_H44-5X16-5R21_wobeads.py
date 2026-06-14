import numpy as np
from XSSModules.cross_section import XcomCrossSection as cs

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
Attention: Although it is called Tire2D, the phantom is actually 3D. 
The phantom is z invariant, i.e. it is the same in XY plane for all Z values.
----------------------------------------------------------------------------'''

def generate_phantom_definition_matrix(energy_kev=70):
    # Maximum number of clipping planes involved
    nclip_max = 4
    nclip = 4*nclip_max
    vclip0 = np.zeros([1,nclip])
    vclip0 = vclip0.flatten()
    vea0 = [1.0,0.0,0.0]
    veb0 = [0.0,1.0,0.0]
    vec0 = [0.0,0.0,1.0]
    vbasis0 = vea0+veb0+vec0
    phantom_def_matrix = []
    
    # Tire dimensions
    W = 419.0 # width of the tire in mm (Height)
    T = 298.0 # thickness of the tire in mm: (Douter - Dinner)/2
    
    rrubber_out = W/2.0
    rrubber_inner = T*1.2/2.0
    rcylinder_length = 500.0 # length of the cylinder in mm
    rgroove = [8.0,12.0,8.0]
    
    vcenter = [0.0, 0.0, 0.0] # center of the tire in mm
    
    # Phantom material
    fn_aramid = 'Aramid.mix'
    fn_nylon = 'Nylon66.mix'
    fn_rubber = 'NaturalRubber.mix'
    fn_steel = '26_Fe.ele'
    fn_air = 'Air.mix'
    
    scale_aramid = 1.0
    scale_nylon = 1.0
    scale_rubber = 1.0
    scale_steel = 1.0
    scale_air = 1.0
    
    aramid_info = cs(fn_aramid)
    mu_aramid = aramid_info.interp_LAC(energy_kev)
    mu_aramid = mu_aramid[0]*scale_aramid
    dict_aramid = {"material": fn_aramid,"scale": scale_aramid}
    
    nylon_info = cs(fn_nylon)
    mu_nylon = nylon_info.interp_LAC(energy_kev)
    mu_nylon = mu_nylon[0]*scale_nylon
    dict_nylon = {"material": fn_nylon,"scale": scale_nylon}
    
    rubber_info = cs(fn_rubber)
    mu_rubber = rubber_info.interp_LAC(energy_kev)
    mu_rubber = mu_rubber[0]*scale_rubber
    dict_rubber = {"material": fn_rubber,"scale": scale_rubber}
    
    steel_info = cs(fn_steel)
    mu_steel = steel_info.interp_LAC(energy_kev)
    mu_steel = mu_steel[0]*scale_steel
    dict_steel = {"material": fn_steel,"scale": scale_steel}
    
    air_info = cs(fn_air)
    mu_air = steel_info.interp_LAC(energy_kev)
    mu_air = mu_air[0]*scale_air
    dict_air = {"material": fn_air,"scale": scale_air}
    
    material_list = []
    
       
    # -------------------------------------------------------------------------
    # [ oID LAC idxPre [vx0] [ra,rb,rc] [vea] [veb] [vec] [d1 ve1] [d2 ve2] ]
    # -------------------------------------------------------------------------
    # Base objects:  Breast + skin
    # -------------------------------------------------------------------------
    # Object 0: Carcass - Rubber - Circular Cylinder
    vprim = [2,   mu_rubber, 0.0] + vcenter + [rrubber_out, rrubber_out, rcylinder_length]
    vprim = np.hstack([vprim,vbasis0])
    
    vclip = [rcylinder_length, 0.0, 0.0,  1.0,
             rcylinder_length, 0.0, 0.0, -1.0,
             T/2.0, 0.0, 1.0,  0.0,
             T/2.0, 0.0, -1.0, 0.0]
   
    vobj = np.hstack([vprim,vclip])
    phantom_def_matrix = vobj
    
    material_list.append(dict_rubber)
    
    # Object 1: Carcass - Air - Circular Cylinder
    vprim = [2,   -mu_rubber, 0.0] + vcenter + [rrubber_inner, rrubber_inner, rcylinder_length]
    vprim = np.hstack([vprim,vbasis0])
    
    vclip = [rcylinder_length, 0.0, 0.0,  1.0,
             rcylinder_length, 0.0, 0.0, -1.0,
             T/3.0, 0.0,  1.0,  0.0,
             T/2.0, 0.0, -1.0, 0.0]
   
    vobj = np.hstack([vprim,vclip])
    phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
    
    material_list.append(dict_rubber)
    
    # Object 2 - 6: Center Groove - Air - Cylinder
    dgroove = 65
    for k in np.arange(-1,3,1):
        vprim = [2,   -mu_rubber, 0.0] + [dgroove*(k-0.5), T/2.0, 0.0] + rgroove
        vprim = np.hstack([vprim,vbasis0])
        
        vclip = [rcylinder_length, 0.0, 0.0,  1.0,
                 rcylinder_length, 0.0, 0.0, -1.0,
                0.0, 0.0, 1.0,  0.0,
                0.0, 0.0, 0.0,  0.0]
    
        vobj = np.hstack([vprim,vclip])
        phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
        
        material_list.append(dict_rubber)
        
    # Object 7-9: casting ply - (clipped cylinder)
    h_inner_clip = T/3.0
    yply_center = T/3.0 + 6.0
    ply_thickness = 1.0
    ply_pitch = 2
    rply = np.sqrt(rrubber_inner**2 - h_inner_clip**2)
    
    for k in np.arange(0,3,1):
        yply_centerk = yply_center + k*ply_pitch
        vprim = [2,   mu_nylon - mu_rubber, 0.0] + [0.0, yply_centerk, 0.0] + [rply, rply, rcylinder_length]
        vprim = np.hstack([vprim,vbasis0])
        
        vclip = [rcylinder_length, 0.0, 0.0,  1.0,
                 rcylinder_length, 0.0, 0.0, -1.0,
                 ply_thickness/2.0, 0.0, 1.0,  0.0,
                 ply_thickness/2.0, 0.0, -1.0, 0.0]
    
        vobj = np.hstack([vprim,vclip])
        phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
        
        material_list.append(dict_nylon)
        
    # Object 10-12: tread reinforcement ply - Aramid (clipped cylinder)
    h_inner_clip = T/3.0
    yply_center = T/3.0 + 12.0
    ply_thickness = 1.0
    ply_pitch = 2
    rply = np.sqrt(rrubber_inner**2 - h_inner_clip**2)
    
    for k in np.arange(0,3,1):
        yply_centerk = yply_center + k*ply_pitch
        vprim = [2,   mu_aramid - mu_rubber, 0.0] + [0.0, yply_centerk, 0.0] + [rply, rply, rcylinder_length]
        vprim = np.hstack([vprim,vbasis0])
        
        vclip = [rcylinder_length, 0.0, 0.0,  1.0,
                 rcylinder_length, 0.0, 0.0, -1.0,
                 ply_thickness/2.0, 0.0, 1.0,  0.0,
                 ply_thickness/2.0, 0.0, -1.0, 0.0]
    
        vobj = np.hstack([vprim,vclip])
        phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
        
        material_list.append(dict_aramid)
    
    # Object 13-19: Low contrast Nylon on sidewall
    rlow_contrast_group = (rrubber_inner + rrubber_out)/2.0
    rlow_contrast = 5 #1
    thetas = np.pi*0.81
    dtheta = np.pi/20.0
        
    for k in np.arange(1,8,1):
        thetak = thetas + k*dtheta#*(1.1**k)
        rlow_contrastk = rlow_contrast#k*rlow_contrast + 3
        vcenter_low_contrast = [rlow_contrast_group*np.cos(thetak), rlow_contrast_group*np.sin(thetak), 0.0]
        vprim = [2,   mu_nylon - mu_rubber, 0.0] + vcenter_low_contrast + [rlow_contrastk, rlow_contrastk, rcylinder_length]
        vprim = np.hstack([vprim,vbasis0])
        
        vclip = [rcylinder_length, 0.0, 0.0,  1.0,
                 rcylinder_length, 0.0, 0.0, -1.0,
                 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0]
    
        vobj = np.hstack([vprim,vclip])
        phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
        
        material_list.append(dict_nylon)
        
        
    # Object 20-25: Low contrast Nylon on tread
    ylow_contrast_group = T/3.0 + 27
    xslow_contrast_group = -rply + 25
    dxlow_contrast = 25
    rlow_contrast = 1
        
    for k in np.arange(0,5,1):
        rlow_contrastk = k*rlow_contrast + 2
        xlow_contrastk = xslow_contrast_group + k*dxlow_contrast
        vcenter_low_contrast = [xlow_contrastk , ylow_contrast_group, 0.0]
        vprim = [2,   mu_nylon - mu_rubber, 0.0] + vcenter_low_contrast + [rlow_contrastk, rlow_contrastk, rcylinder_length]
        vprim = np.hstack([vprim,vbasis0])
        
        vclip = [rcylinder_length, 0.0, 0.0,  1.0,
                 rcylinder_length, 0.0, 0.0, -1.0,
                 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0]
    
        vobj = np.hstack([vprim,vclip])
        phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
        
        material_list.append(dict_nylon)
        
    # Object 26-34: Air spatial resolution paterns Group 1: 1.0 mm thickness, 1.0 mm space, 10 mm height
    xsaircontrast = 20
    dxaircontrast = 15
    raircontrast = 4
    dlpcm = 2.0
    
    for i in np.arange(0,9,1):
        lpcm = 10 + i*dlpcm
        xaircontrast = xsaircontrast + dxaircontrast*i
        dair = 5.0/lpcm        
        for k in np.arange(0,4,1):        
            vcenter_low_contrast = [xaircontrast + k*dair*2.0 , ylow_contrast_group, 0.0]
            vprim = [2,   -mu_rubber, 0.0] + vcenter_low_contrast + [raircontrast, raircontrast, rcylinder_length]
            vprim = np.hstack([vprim,vbasis0])
            
            vclip = [rcylinder_length, 0.0, 0.0,  1.0,
                    rcylinder_length, 0.0, 0.0, -1.0,
                    dair/2.0,  1.0,  0.0, 0.0,
                    dair/2.0, -1.0,  0.0, 0.0]
        
            vobj = np.hstack([vprim,vclip])
            phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
            
            material_list.append(dict_air)
    
    # Object 35-41: Aramid spatial resolution paterns 
    thetas = np.pi*0.135
    dtheta = np.pi/26.0
    dlpcm = 2.0
    
    for i in np.arange(0,9,1):
        lpcm = 10.0 + i*dlpcm
        thetai = thetas - i*dtheta
        vcenter_aramid_contrasti = np.array([rlow_contrast_group*np.cos(thetai), rlow_contrast_group*np.sin(thetai), 0.0])
        vthetaper = np.array([-np.sin(thetai),np.cos(thetai),0.0])
        dair = 5.0/lpcm        
        for k in np.arange(-1,3,1):        
            vcenter_aramid_contrastk = vcenter_aramid_contrasti + k*dair*2.0*vthetaper
            vprim = [2,   mu_aramid-mu_rubber, 0.0] + vcenter_aramid_contrastk.tolist() + [raircontrast, raircontrast, rcylinder_length]
            vprim = np.hstack([vprim,vbasis0])
            
            vclip = [rcylinder_length, 0.0, 0.0,  1.0,
                    rcylinder_length, 0.0, 0.0, -1.0,
                    dair/2.0,   vthetaper[0],   vthetaper[1], 0.0,
                    dair/2.0,  -vthetaper[0],  -vthetaper[1], 0.0]
        
            vobj = np.hstack([vprim,vclip])
            phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
            
            material_list.append(dict_air)

        
    # # Object 42: Single large bead 
    # rlow_contrast_group = (rrubber_inner + rrubber_out)/2.0
    # rbead_large = 12
    # theta = np.pi*1.23
        
    # vcenter_bead = [rlow_contrast_group*np.cos(theta), rlow_contrast_group*np.sin(theta), 0.0]
    # vprim = [2,   mu_steel - mu_rubber, 0.0] + vcenter_bead + [rbead_large, rbead_large, rcylinder_length]
    # vprim = np.hstack([vprim,vbasis0])
        
    # vclip = [rcylinder_length, 0.0, 0.0,  1.0,
    #         rcylinder_length, 0.0, 0.0, -1.0,
    #         0.0, 0.0, 0.0, 0.0,
    #         0.0, 0.0, 0.0, 0.0]
    
    # vobj = np.hstack([vprim,vclip])
    # phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
        
    # material_list.append(dict_steel)
    
    # # Object 43: Single large bead 
    # rlow_contrast_group = (rrubber_inner + rrubber_out)/2.0
    # rbead_large = 12
    # theta = -np.pi*0.23
        
    # vcenter_bead = [rlow_contrast_group*np.cos(theta), rlow_contrast_group*np.sin(theta), 0.0]
    # vprim = [2,   mu_steel - mu_rubber, 0.0] + vcenter_bead + [rbead_large, rbead_large, rcylinder_length]
    # vprim = np.hstack([vprim,vbasis0])
        
    # vclip = [rcylinder_length, 0.0, 0.0,  1.0,
    #         rcylinder_length, 0.0, 0.0, -1.0,
    #         0.0, 0.0, 0.0, 0.0,
    #         0.0, 0.0, 0.0, 0.0]
    
    # vobj = np.hstack([vprim,vclip])
    # phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
        
    # material_list.append(dict_steel)
    
    # Object 28-end: a group of small beads 
    # rlow_contrast_group = (rrubber_inner + rrubber_out)/2.0
    # rbead_small = 0.4
    # dbead_small = rbead_small*2.0
    # theta = -np.pi*0.25
    # nlayer = 8
    # for k in np.arange(0,nlayer,1):
    #     rgroupk = k*dbead_small
    #     group_circumference = 2.0*np.pi*rgroupk
    #     nbeads = int(np.floor(group_circumference/dbead_small))
    #     nbeads = np.maximum(nbeads, 1)  # Ensure at least 3 beads in the group
    #     dtheta = 2.0*np.pi/nbeads
    #     for j in np.arange(0,nbeads,1):
    #         thetak = j*dtheta
    #         vcenter_beadk = [rlow_contrast_group*np.cos(theta)+rgroupk*np.cos(thetak), rlow_contrast_group*np.sin(theta)+rgroupk*np.sin(thetak), 0.0]
    #         vprim = [2,   mu_steel - mu_rubber, 0.0] + vcenter_beadk + [rbead_small, rbead_small, rcylinder_length]
    #         vprim = np.hstack([vprim,vbasis0])
            
    #         vclip = [rcylinder_length, 0.0, 0.0,  1.0,
    #                  rcylinder_length, 0.0, 0.0, -1.0,
    #                  0.0, 0.0, 0.0, 0.0,
    #                  0.0, 0.0, 0.0, 0.0]
    
    #         vobj = np.hstack([vprim,vclip])
    #         phantom_def_matrix = np.vstack([phantom_def_matrix,vobj])
            
    #         material_list.append(dict_steel)
            
    
    return phantom_def_matrix, material_list
    
    
 


