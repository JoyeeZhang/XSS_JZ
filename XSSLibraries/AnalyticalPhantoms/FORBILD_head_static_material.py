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
----------------------------------------------------------------------------'''
def generate_phantom_definition_matrix(energy_kev=70):
    # Maximum number of clipping planes involved
    nclip_max = 2
    nclip = 4*nclip_max
    vclip0 = np.zeros([1,nclip])
    vclip0 = vclip0.flatten()
    vbasis = np.reshape(np.eye(3),[1,9])
    vbasis = vbasis.flatten()
    phantom_def_matrix = []
    
    # Angles
    deg15  =  15*np.pi/180
    deg120 = 120*np.pi/180
    deg30  =  30*np.pi/180
    deg150 = 150*np.pi/180
    deg58  =58.1*np.pi/180
    deg60  =  60*np.pi/180
    
    # -------------------------------------------------------------------------
    # >>>>>> Generate phantom materials
    # -------------------------------------------------------------------------
    fn_bone = "Bone(ICRU-44).mix"
    fn_brain = "Brain.mix"
    fn_eye = "EyeLens.mix"
    fn_air = "Air.mix"
    
    
    material_list = [
        {"material": fn_eye,    "scale": 1.0}, #Object 1: Eyes
        {"material": fn_eye,    "scale": 1.0}, #Object 2: Eyes
        {"material": fn_brain,  "scale": 1.0525/1.05}, #Object 3: Lesion
        {"material": fn_brain,  "scale": 1.0475/1.05}, #Object 4: Lesion        
        {"material": fn_bone,   "scale": 1.0}, #Object 5: Calotte
        {"material": fn_brain,  "scale": 1.0},  #Object 6: Brain matter
        {"material": fn_air,    "scale": 1.0}, #Object 7: Frontal sinus
        {"material": fn_bone,   "scale": 1.0}, #Object 8: Bone @ frontal sinus
        {"material": fn_bone,   "scale": 1.0}, #Object 9: Bone @ frontal sinus
        {"material": fn_bone,   "scale": 1.0}, #Object 10: Bone @ frontal sinus
        {"material": fn_bone,   "scale": 1.0}, #Object 11: Bone @ frontal sinus
        {"material": fn_brain,  "scale": 1.045/1.05}, #Object 12: Ventricle
        {"material": fn_brain,  "scale": 1.055/1.05}, #Object 13: Subdural hematoma
        {"material": fn_bone,   "scale": 1.0}, #Object 14: Bone @ frontal sinus
        {"material": fn_bone,   "scale": 1.0}, #Object 15: Bone @ frontal sinus
        {"material": fn_bone,   "scale": 1.0}, #Object 16: Internal occipital protuberance
        {"material": fn_bone,   "scale": 1.0}  #Object 17: Internal occipital protuberance
    ]
    
    num_obj = len(material_list)
    LAC = np.zeros(num_obj, dtype=np.float64)
    
    for k in range(num_obj):
            materialk = material_list[k]
            material_name = materialk['material']
            scale = np.float32(materialk['scale'])
            
            materialk_info = cs(material_name)
            LACk_mm = materialk_info.interp_LAC(energy_kev)
            LAC[k] = LACk_mm*scale 
    
    
    # -------------------------------------------------------------------------
    # >>>>>> Generate phantom matrix
    # -------------------------------------------------------------------------
    # [ oID LAC idxPre [vx0] [ra,rb,rc] [vea] [veb] [vec] [d1 ve1] [d2 ve2] ]
    # -------------------------------------------------------------------------
    # Precedence objects - 5 objects including (5,6,7,16,17)
    # ---- Object 5: Calotte (1.8)
    lac5  = LAC[4]#1.8
    vprim = [1,   lac5, 1,            # ellipsoid, LAC, precedence index
             0,   0,    0,            # center of the object
             96,  120,  125]          # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])
    v5 = np.hstack([vprim,vclip0])
    phantom_def_matrix = v5
    
    # # ---- Object 6: Brain matter (1.05)
    lac6  = LAC[5]#1.05
    vprim = [1,   lac6,  1,           # ellipsoid, LAC, precedence index
             0,   0,     0,           # center of the object
             90,  114,  119]          # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])
    v6 = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,v6])
    
    # # ---- Object 7: Frontal sinus (0)
    lac7  = LAC[6]#0.0
    vprim = [1,   lac7,  1,           # ellipsoid, LAC, precedence index
             0,   84,    0,           # center of the object
             18,  30,   30]           # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])
    v7 = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,v7])
    
    # ---- Object 16: Internal occipital protuberance (1.8)
    lac16 = LAC[15]#1.8
    vprim = [3,   lac16,  1,          # paraboloid, LAC, precedence index
             0,   -94,   -2,          # center of the object
             2,   2,    10]           # [ra,rb,rc]
    vbasis16 = [1,0,0,0,0,1,0,-1,0]   # [vea,veb,vec] w/ vec as axis
    vprim = np.hstack([vprim,vbasis16])
    
    vclip16  = [-10, 0,  1, 0,
                 25, 0, -1, 0]
    v16 = np.hstack([vprim,vclip16])
    
    phantom_def_matrix = np.vstack([phantom_def_matrix,v16])
    
    # ---- Object 17: Internal occipital protuberance (1.8)
    lac17 = LAC[16]#1.8
    vprim = [3,   lac17,  1,           # paraboloid, LAC, precedence index
             0,   -94,    2,           # center of the object
             2,   2,     10]           # [ra,rb,rc]
    vbasis17 = [1,0,0,0,0,1,0,-1,0]    # [vea,veb,vec] w/ vec as axis
    vprim = np.hstack([vprim,vbasis17])
    
    vclip17  = [-10, 0,  1, 0,
                 25, 0, -1, 0]
    v17 = np.hstack([vprim,vclip17])
    
    phantom_def_matrix = np.vstack([phantom_def_matrix,v17])
    
    # -------------------------------------------------------------------------
    # Additive objects
    # ---- Object 1: Eye (1.06)
    lac1 = LAC[0]#1.06
    vprim = [0,   lac1 - lac6,   0,    # sphere, LAC, precedence index
            -47,  43,     8.72,        # center of the object
             20,  20,     20]          # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])
    v1 = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,v1])
    
    # ---- Object 2: Eye (1.06)
    lac2 = LAC[1]#1.06
    vprim = [0,   lac2 - lac6,   0,    # sphere, LAC, precedence index
             47,  43,    8.72,         # center of the object
             20,  20,    20]           # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])
    v2 = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,v2])
    
    # ---- Object 3: lesion (1.0525)
    lac3 = LAC[2]#1.0525
    vprim = [0,   lac3 - lac6, 0,    # sphere, LAC, precedence index
            -10.8, -90,    0.0,        # center of the object
             4,     4,     4]          # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])
    v3 = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,v3])
    
    # ---- Object 4: Lesion (1.0475)
    lac4 = LAC[3]#1.0475
    vprim = [0,   lac4 - lac6, 0,    # sphere, LAC, precedence index
             10.8, -90,     0.0,       # center of the object
             4, 4,    4]               # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])
    v4 = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,v4])
    
    # ---- Object 8: bone @ frontal sinus (1.8)
    lac8 = LAC[7]#1.8
    vea8 = [np.cos(deg15)*np.cos(deg120), np.sin(deg120),-np.sin(deg15)*np.cos(deg120)]
    vec8 = [np.sin(deg15),0.0,np.cos(deg15)]
    veb8 = np.cross(vec8,vea8)
    vbasis8 = np.hstack([vea8,veb8,vec8])  
    
    vprim = [1,   lac8 - lac6, 0,        # ellipsoid, LAC, precedence index
            -19,  54,      0.0,         # center of the object
             12*np.cos(deg15), 4.2*np.cos(deg15), 30]  # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis8])
    
    v8 = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,v8])   
    
    # ---- Object 9: bone @ frontal sinus (1.8)
    lac9 = LAC[8]#1.8
    vea9  = [np.cos(deg15)*np.cos(deg60), np.sin(deg60),np.sin(deg15)*np.cos(deg60)]
    vec9  = [np.sin(-deg15),0.0,np.cos(deg15)]
    veb9  = np.cross(vec9,vea9)
    vbasis9 = np.hstack([vea9,veb9,vec9])
    
    vprim = [1,   lac9 - lac6, 0,         # ellipsoid, LAC, precedence index
             19,  54,      0.0,          # center of the object
             12*np.cos(deg15), 4.2*np.cos(deg15), 30]  # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis9])
    
    v9 = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,v9])
    
    # ---- Object 10: bone @ frontal sinus (1.8)
    lac10 = LAC[9]#1.8
    vea10 = [np.cos(deg150),np.sin(deg150),0]
    vec10 = [0,0,1]
    veb10 = np.cross(vec10,vea10)
    vbasis10 = np.hstack([vea10,veb10,vec10])
    
    rc10 = 40*0.5
    vprim = [2,   lac10 - lac6, 0,          # cylinder, LAC, precedence index
            -43,  68,       -10,          # center of the object
             18, 2.4,   rc10]             # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis10])
    
    vclip10 = [rc10,0,0, 1.0,
               rc10,0,0,-1.0]
    v10 = np.hstack([vprim,vclip10])
    phantom_def_matrix = np.vstack([phantom_def_matrix,v10])
    
    # ---- Object 11: bone @ frontal sinus (1.8)
    lac11 = LAC[10]#1.8
    vea11 = [np.cos(deg30),np.sin(deg30),0]
    vec11 = [0,0,1] 
    veb11 = np.cross(vec11,vea11)
    vbasis11 = np.hstack([vea11,veb11,vec11])
    
    rc11 = 40*0.5
    vprim = [2,   lac11 - lac6, 0,           # cylinder, LAC, precedence index
             43,  68,       -10,           # center of the object
             18,  2.4,    rc11]            # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis11])

    vclip11 = [rc11,0,0, 1.0,
               rc11,0,0,-1.0]
    v11 = np.hstack([vprim,vclip11])
    phantom_def_matrix = np.vstack([phantom_def_matrix,v11])
    
    # ---- Object 12: ventricle (1.045)
    lac12 = LAC[11]#1.045
    vprim = [1,   lac12 - lac6, 0,          # ellipsoid, LAC, precedence index
             0,   -36,         0,           # center of the object
             18,   36,        36]           # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis])
    v12 = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,v12])
    
    # ---- Object 13: subdural hematoma (1.055)
    lac13 = LAC[12]#1.055
    vea13 = [np.cos(deg58), np.sin(deg58),0]
    vec13 = [0,0,1]
    veb13 = np.cross(vec13,vea13)
    vbasis13 = np.hstack([vea13,veb13,vec13])
    
    vprim = [1,   lac13 - lac6, 0,           # ellipsoid, LAC, precedence index
             63.93945,-63.93945,0,           # center of the object
             12, 4.2,       12]              # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis13])
    
    v13 = np.hstack([vprim,vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix,v13])
    
    # ---- Object 14: bone @ frontal sinus (1.8)    
    lac14 = LAC[13]#1.8
    veb14 = [0, np.sin(deg30),np.cos(deg30)]
    vec14 = [0,-np.cos(deg30),np.sin(deg30)]
    vea14 = np.cross(veb14,vec14)
    vbasis14 = np.hstack([vea14,veb14,vec14])
    
    rc14 = 5*np.cos(deg15)*0.5
    vprim = [2,   lac14 - lac6,  0,              # cylinder, LAC, precedence index
             0,   36,          0,              # center of the object
             12,  40,       rc14]              # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis14])
    
    vclip14 = [rc14,0,-np.cos(deg30), np.sin(deg30),
               rc14,0, np.cos(deg30),-np.sin(deg30)]
    v14 = np.hstack([vprim,vclip14])
    phantom_def_matrix = np.vstack([phantom_def_matrix,v14])
    
    # ---- Object 15: bone @ frontal sinus (1.8)    
    lac15 = LAC[14]#1.8
    veb15 = [0,-np.sin(deg30), np.cos(deg30)]
    vec15 = [1, 0, 0]
    vea15 = np.cross(veb15,vec15)
    vbasis15 = np.hstack([vea15,veb15,vec15])
    
    rc15  = 4*0.5
    vprim = [2,   lac15 - lac7,   0,            # cylinder, LAC, precedence index
             0,   96,          0,             # center of the object
             5.25561, 20,  rc15]              # [ra,rb,rc]
    vprim = np.hstack([vprim,vbasis15])
    
    vclip15 = [rc15, 1,0,0,
               rc15,-1,0,0]
    v15 = np.hstack([vprim,vclip15])
    phantom_def_matrix = np.vstack([phantom_def_matrix,v15])
    
    return phantom_def_matrix, material_list
    
    
 


