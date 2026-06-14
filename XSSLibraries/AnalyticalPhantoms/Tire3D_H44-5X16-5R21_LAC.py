import numpy as np
from XSSModules.cross_section import XcomCrossSection as cs

'''----------------------------------------------------------------------------
[F] generate_phantom_definition_matrix():   
[O] phantom_def_matrix: [nobj,ncol]
        each row defines 1 object as described below
        [ oID LAC idxPre [vx0] [ra,rb,rc] [vea] [veb] [vec] [d1 ve1] [d2 ve2] ]
        oID: shape ID
        LAC: pixel value
        idxPre: 1 - precedence (always at the beginning of the matrix)
                0 - additive
        vx0: object center along the basis vectors defined below
        [ra,rb,rc]: half axes along the basis vectors
                    for torus, ra represents tube radius, rb represents torus radius, and rc equals 0
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
        torus:        6
    Precedence rule
        Precedence is indicated by 1
        Always start with object with precedence rule
        The precedence objects are always listed with an ascending priority
    Object with precedence rule
        None
    Unit: mm
----------------------------------------------------------------------------'''

def generate_phantom_definition_matrix(energy_kev = 70):

    # -------------------------------------------------------------------------
    # Phantom material
    # -------------------------------------------------------------------------
    fn_aramid = 'Aramid.mix'
    fn_nylon = 'Nylon66.mix'
    fn_rubber = 'TreadRubberA.mix'
    fn_steel = '26_Fe.ele'
    fn_air = 'Air.mix'

    scale_aramid = 1.0
    scale_nylon = 1.0
    scale_rubber = 1.0
    scale_steel = 1.0
    scale_air = 1.0

    aramid_info = cs(fn_aramid)
    mu_aramid = aramid_info.interp_LAC(energy_kev)
    mu_aramid = mu_aramid[0] * scale_aramid
    dict_aramid = {"material": fn_aramid, "scale": scale_aramid}

    nylon_info = cs(fn_nylon)
    mu_nylon = nylon_info.interp_LAC(energy_kev)
    mu_nylon = mu_nylon[0] * scale_nylon
    dict_nylon = {"material": fn_nylon, "scale": scale_nylon}

    rubber_info = cs(fn_rubber)
    mu_rubber = rubber_info.interp_LAC(energy_kev)
    mu_rubber = mu_rubber[0] * scale_rubber
    dict_rubber = {"material": fn_rubber, "scale": scale_rubber}

    steel_info = cs(fn_steel)
    mu_steel = steel_info.interp_LAC(energy_kev)
    mu_steel = mu_steel[0] * scale_steel
    dict_steel = {"material": fn_steel, "scale": scale_steel}

    air_info = cs(fn_air)
    mu_air = steel_info.interp_LAC(energy_kev)
    mu_air = mu_air[0] * scale_air
    dict_air = {"material": fn_air, "scale": scale_air}

    material_list = []

    # -------------------------------------------------------------------------
    # Clipping plane setup
    # -------------------------------------------------------------------------
    nclip_max = 2
    nclip = 4 * nclip_max
    vclip0 = np.zeros([1, nclip])
    vclip0 = vclip0.flatten()

    # -------------------------------------------------------------------------
    # Coordinate system
    # -------------------------------------------------------------------------
    vea = [1, 0, 0]
    veb = [0, 1, 0]
    vec = [0, 0, 1]
    vbasis0 = vea + veb + vec

    # -------------------------------------------------------------------------
    # Global tire geometry parameters
    # -------------------------------------------------------------------------
    W = 419.0  # tire width
    T = 298.0  # tire thickness
    R = 831.5 / 2 # wheel radius (main torus radius)
    out_r = W / 2  # outer tube radius
    out_thickness = 50.0 # tread thickness (outer rubber)
    in_thickness = 80.0 # inner thickness
    width = 2 * np.sqrt((out_r) ** 2 - (T / 2) ** 2) # tread width

    # -------------------------------------------------------------------------
    # [ oID LAC idxPre [vx0] [ra,rb,rc] [vea] [veb] [vec] [d1 ve1] [d2 ve2] ]
    # -------------------------------------------------------------------------
    # Precedence objects - 0 objects
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Additive objects
    # -------------------------------------------------------------------------
    # ---- Object group 0: Tire body
    # ---- Object 0-0 & 0-1: Outer tread + Inner hollow
    # Object 0-0 Outer
    lac0 = mu_rubber
    lac00 = lac0
    vprim = [6, lac00, 0,   # torus, LAC, precedence index
             0, 0, 0,       # center of the object
             out_r, R, 0]   # [r, R, 0]
    vbasis00 = vbasis0
    vprim = np.hstack([vprim, vbasis00])
    vclip00 = [-width/2] + vec + [0.0, 0.0, 0.0, 0.0]
    v00 = np.hstack([vprim, vclip00])
    phantom_def_matrix = v00
    # Object 0-1 Inner
    lac01 = - lac0
    in_R = R + (in_thickness - out_thickness) / 2
    in_r = np.sqrt((np.sqrt(out_r ** 2 - (width / 2) ** 2) - out_thickness - (in_R - R)) ** 2 + (width / 2) ** 2)
    vprim = [6, lac01, 0,   # torus, LAC, precedence index
             0, 0, 0,       # center of the object
             in_r, in_R, 0] # [r, R, 0]
    vbasis01 = vbasis0
    vprim = np.hstack([vprim, vbasis01])
    vclip01 = [-width/2] + vec + [0.0, 0.0, 0.0, 0.0]
    v01 = np.hstack([vprim, vclip01])
    phantom_def_matrix = np.vstack([phantom_def_matrix, v01])
    # ---- Object 0-2 & 0-3: Outer tread + Inner hollow (symmetric duplication)
    # Object 0-2 Outer
    lac02 = mu_rubber
    vprim = [6, lac02, 0,   # torus, LAC, precedence index
             0, 0, 0,       # center of the object
             out_r, R, 0]   # [r, R, 0]
    vbasis02 = vbasis0
    vprim = np.hstack([vprim, vbasis02])
    vclip02 = [-width/2] + [-x for x in vec] + [0.0, 0.0, 0.0, 0.0]
    v02 = np.hstack([vprim, vclip02])
    phantom_def_matrix = np.vstack([phantom_def_matrix, v02])
    # Object 0-3 Inner
    lac03 = - lac02
    vprim = [6, lac03, 0,   # torus, LAC, precedence index
             0, 0, 0,       # center of the object
             in_r, in_R, 0] # [r, R, 0]
    vbasis03 = vbasis0
    vprim = np.hstack([vprim, vbasis03])
    vclip03 = [-width/2] + [-x for x in vec] + [0.0, 0.0, 0.0, 0.0]
    v03 = np.hstack([vprim, vclip03])
    phantom_def_matrix = np.vstack([phantom_def_matrix, v03])
    # ---- Object 0-4 & 0-5: Flatten bottom + Inner hollow
    # Object 0-4 Outer
    lac04 = mu_rubber
    ra_out = R + np.sqrt(out_r**2 - (width/2)**2)
    vprim = [2, lac04, 0,           # cylinder, LAC, precedence index
             0, 0, 0,               # center of the object
             ra_out, ra_out, width] # [ra, rb, rc]
    vbasis04 = vbasis0
    vprim = np.hstack([vprim, vbasis04])
    vclip04 = [width/2] + [-x for x in vec] + [width / 2] + vec
    v04 = np.hstack([vprim, vclip04])
    phantom_def_matrix = np.vstack([phantom_def_matrix, v04])
    # Object 0-5 Inner
    lac05 = - lac04
    ra_in = ra_out - out_thickness
    vprim = [2, lac05, 0,           # cylinder, LAC, precedence index
             0, 0, 0,               # center of the object
             ra_in, ra_in, width]   # [ra, rb, rc]
    vbasis05 = vbasis0
    vprim = np.hstack([vprim, vbasis05])
    vclip05 = [width / 2] + [-x for x in vec] + [width / 2] + vec
    v05 = np.hstack([vprim, vclip05])
    phantom_def_matrix = np.vstack([phantom_def_matrix, v05])
    # ---- Object 0-6 & 0-7: beads (connection bar)
    # Object 6 Outer tread + inner hollow
    lac06 = mu_rubber
    bar_thickness = 20
    theta_down = (np.pi / 2 - np.arccos((width / 2.0) / in_r))
    theta_up = (np.pi / 2 - np.arccos((width / 2.0) / out_r))
    ira06_down = in_R - np.sqrt(in_r ** 2 - (width / 2.0) ** 2)
    ira06_up = R - np.sqrt(out_r ** 2 - (width / 2.0) ** 2)
    irc06_down = ira06_down / np.tan(theta_down)
    irc06_up = ira06_up / np.tan(theta_up)
    ic06_down = - irc06_down + (width / 2.0)
    ic06_up = - irc06_up + (width / 2.0)
    vbasis06 = vbasis0
    # Outer one
    vprim06_down = ([3, lac06, 0] +                         # paraboloid, LAC, precedence index
                    [ic06_down * x for x in vec] +          # center of the object
                    [ira06_down, ira06_down, irc06_down])   # [ra, rb, rc]
    vclip06_down = [irc06_down] + vec + [bar_thickness - irc06_down] + [-x for x in vec]
    vprim06_down = np.hstack([vprim06_down, vbasis06])
    v06_down = np.hstack([vprim06_down, vclip06_down])
    phantom_def_matrix = np.vstack([phantom_def_matrix, v06_down])
    # Inner one
    vprim06_up = ([3, - lac06, 0] +                         # paraboloid, LAC, precedence index
                  [ic06_up * x for x in vec] +              # center of the object
                  [ira06_up, ira06_up, irc06_up])           # [ra, rb, rc]
    vclip06_up = [irc06_up] + vec + [bar_thickness - irc06_up] + [-x for x in vec]
    vprim06_up = np.hstack([vprim06_up, vbasis06])
    v06_up = np.hstack([vprim06_up, vclip06_up])
    phantom_def_matrix = np.vstack([phantom_def_matrix, v06_up])
    # Object 7 Outer tread + inner hollow (symmetric duplication)
    lac07 = lac06
    ira07_down = ira06_down
    ira07_up = ira06_up
    irc07_down = irc06_down
    irc07_up = irc06_up
    ic07_down = - ic06_down
    ic07_up = - ic06_up
    vbasis07 = vbasis0
    # Outer one
    vprim07_down = ([3, lac07, 0] +                         # paraboloid, LAC, precedence index
                    [ic07_down * x for x in vec] +          # center of the object
                    [ira07_down, ira07_down, irc07_down])   # [ra, rb, rc]
    vclip07_down = [irc07_down] + [-x for x in vec] + [bar_thickness - irc07_down] + vec
    vprim07_down = np.hstack([vprim07_down, vbasis07])
    v07_down = np.hstack([vprim07_down, vclip07_down])
    phantom_def_matrix = np.vstack([phantom_def_matrix, v07_down])
    # Inner one
    vprim07_up = ([3, - lac07, 0] +                         # paraboloid, LAC, precedence index
                  [ic07_up * x for x in vec] +              # center of the object
                  [ira07_up, ira07_up, irc07_up])           # [ra, rb, rc]
    vclip07_up = [irc07_up] + [-x for x in vec] + [bar_thickness - irc07_up] + vec
    vprim07_up = np.hstack([vprim07_up, vbasis07])
    v07_up = np.hstack([vprim07_up, vclip07_up])
    phantom_def_matrix = np.vstack([phantom_def_matrix, v07_up])
    # ---- Object 0-8 & 0-9: beads (tail)
    # Object 0-8
    lac08 = mu_rubber
    bar_width = (irc06_down - bar_thickness) * np.tan(theta_down) - (irc06_up - bar_thickness) * np.tan(theta_up)
    r08 = (bar_width / 2.0) / np.cos(theta_down)
    R08 = (irc06_up - bar_thickness) * np.tan(theta_up) + r08 * np.cos(theta_down)
    height08 = width / 2.0 - bar_thickness + r08 * np.sin(theta_down)
    vprim = ([6, lac08, 0] +                    # torus, LAC, precedence index
             [height08 * x for x in vec] +      # center of the object
             [r08, R08, 0])                     # [r, R, 0]
    vbasis08 = vbasis0
    vprim = np.hstack([vprim, vbasis08])
    vclip08 = [- r08 * np.sin(theta_down)] + vec + [0.0, 0.0, 0.0, 0.0]
    v08 = np.hstack([vprim, vclip08])
    phantom_def_matrix = np.vstack([phantom_def_matrix, v08])
    # Object 0-9
    lac09 = lac08
    r09 = r08
    R09 = R08
    height09 = height08
    vprim = ([6, lac09, 0] +                    # torus, LAC, precedence index
             [- height09 * x for x in vec] +    # center of the object
             [r09, R09, 0])                     # [r, R, 0]
    vbasis09 = vbasis0
    vprim = np.hstack([vprim, vbasis09])
    vclip09 = [- r09 * np.sin(theta_down)] + [- x for x in vec] + [0.0, 0.0, 0.0, 0.0]
    v09 = np.hstack([vprim, vclip09])
    phantom_def_matrix = np.vstack([phantom_def_matrix, v09])

    # ---- Object 1: nylon66 (top two rows) / aramid (bottom two rows)
    lac10 = mu_nylon - mu_rubber # nylon66
    lac11 = mu_aramid - mu_rubber # aramid
    group_num = 4
    # define size and position of each object group
    size_array = np.array([0.6, 0.8, 1.0, 1.2]) / 2
    row_num = 10
    column_num = 4
    rgap = 4 * size_array
    cgap = 2 * np.sqrt(3) * size_array
    first_height = np.zeros((group_num,))
    first_height[0] = 0.9 * width / 2
    for g in range(group_num):
        first_height[g] = 30 * (group_num - g) + 5
    first_R = np.zeros((group_num,))
    first_R[0] = ra_in + 0.1 * (ra_out - ra_in)
    for g in range(group_num-1):
        first_R[g+1] = first_R[g] + column_num*cgap[g]
    vbasis1 = vbasis0
    # generate object groups
    for g in range(group_num):
        irgap = rgap[g]
        icgap = cgap[g]
        for r in range(row_num):
            for c in range(column_num):
                if c < column_num//2:
                    lac = lac11
                else:
                    lac = lac10
                if c % 2 == 0:
                    iheight = first_height[g] + r*irgap - irgap/2
                else:
                    iheight = first_height[g] + r*irgap
                iR = first_R[g] + c*icgap
                vprim = ([6, lac, 0] +                  # torus, LAC, precedence
                         [-iheight*x for x in vec] +    # center of the object
                         [size_array[g], iR, 0])        # [r, R, 0]
                vprim = np.hstack([vprim, vbasis1])
                v1 = np.hstack([vprim, vclip0])
                phantom_def_matrix = np.vstack([phantom_def_matrix, v1])

    # ---- Object 2: air (sixteen groups of air bar patterns)
    lac2 = - mu_rubber
    group_num = 16
    # define size and position of each object group
    xsaircontrast = 35
    dxaircontrast = 7
    raircontrast = 6
    dlpcm = 2.90
    R2 = (ra_in + ra_out)/2
    vbasis2 = vbasis0
    for g in range(group_num):
        lpcm = 8 + g*dlpcm
        xaircontrast = xsaircontrast + dxaircontrast*g
        dair = 5.0/lpcm
        for i in range(4):
            ##### torus version #####
            # pos = - (xaircontrast + i*dair*2.0)
            # vprim = [6, lac2, 0] + [-pos*x for x in vec] + [raircontrast, R2, 0]
            # vprim = np.hstack([vprim, vbasis2])
            # vclip2 = [dair/2] + [-x for x in vec] + [dair/2] + vec
            # v2 = np.hstack([vprim, vclip2])
            # phantom_def_matrix = np.vstack([phantom_def_matrix, v2])
            ##### cylinder version #####
            # outer
            pos = - (xaircontrast + i*dair*2.0)
            vprim20 = ([2, lac2, 0] +                               # cylinder, LAC, precedence
                       [-pos*x for x in vec] +                      # center of the object
                       [R2+raircontrast, R2+raircontrast, dair])    # [ra, rb, rc]
            vprim20 = np.hstack([vprim20, vbasis2])
            vclip20 = [dair/2] + [-x for x in vec] + [dair/2] + vec
            v20 = np.hstack([vprim20, vclip20])
            phantom_def_matrix = np.vstack([phantom_def_matrix, v20])
            # inner
            vprim21 = [2, -lac2, 0] + [-pos*x for x in vec] + [R2-raircontrast, R2-raircontrast, dair]
            vprim21 = np.hstack([vprim21, vbasis2])
            vclip21 = [dair/2] + [-x for x in vec] + [dair/2] + vec
            v21 = np.hstack([vprim21, vclip21])
            phantom_def_matrix = np.vstack([phantom_def_matrix, v21])

    # ---- Object 3: aramid (top three layers) / nylon 66 (bottom three layers)
    lac30 = mu_aramid - mu_rubber # aramid
    lac31 = mu_nylon - mu_rubber # nylon66
    # define size and position of each object group
    gap = 2
    thickness = 1
    vbasis3 = vbasis0
    height3 = 30
    group_num = 6
    for i in range(group_num):
        ra3 = (ra_in + ra_out)/2 - 10 - i*gap
        if i < group_num/2:
            lac = lac30
        else:
            lac = lac31
        # outer
        vprim30 = [2, lac, 0,                                   # cylinder, LAC, precedence index
                   0, 0, 0,                                     # center of the object
                   ra3+thickness/2, ra3+thickness/2, height3]   # [ra, rb, rc]
        vprim30 = np.hstack([vprim30, vbasis3])
        vclip30 = [height3/2] + [-x for x in vec] + [height3/2] + vec
        v30 = np.hstack([vprim30, vclip30])
        phantom_def_matrix = np.vstack([phantom_def_matrix, v30])
        # inner
        vprim31 = [2, -lac, 0,                                  # cylinder, LAC, precedence index
                   0, 0, 0,                                     # center of the object
                   ra3-thickness/2, ra3-thickness/2, height3]   # [ra, rb, rc]
        vprim31 = np.hstack([vprim31, vbasis3])
        vclip31 = vclip30.copy()
        v31 = np.hstack([vprim31, vclip31])
        phantom_def_matrix = np.vstack([phantom_def_matrix, v31])

    # ---- Object 4: nylon 66
    lac4 = mu_nylon - mu_rubber
    R4 = (ra_in + ra_out) / 2
    r4 = 8
    height4 = 20
    vprim = ([6, lac4, 0] +                 # torus, LAC, precendence index
             [-height4*x for x in vec] +    # center of the object
             [r4, R4, 0])                   # [ra, rb, rc]
    vbasis4 = vbasis0
    vprim = np.hstack([vprim, vbasis4])
    v4 = np.hstack([vprim, vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix, v4])

    # ---- Object 5: aramid
    lac5 = mu_aramid - mu_rubber
    R5 = R4
    r5 = r4
    height5 = -height4
    vprim = ([6, lac5, 0] +                 # torus, LAC, precedence index
             [-height5*x for x in vec] +    # center of the object
             [r5, R5, 0])                   # [ra, rb, rc]
    vbasis5 = vbasis0
    vprim = np.hstack([vprim, vbasis5])
    v5 = np.hstack([vprim, vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix, v5])

    # ---- Object 6: nylon (nine groups of bar patterns)
    lac6 = mu_nylon - mu_rubber
    rad6 = np.deg2rad(np.array([28, 20, 12, 4]))
    group_num = 2*rad6.shape[0]
    # define size and position of each object group
    r_mid = (in_r + out_r) / 2
    length6 = 8
    dlpcm = 4.0
    width6 = 5.0 / (8.0 + dlpcm*np.arange(group_num))
    gap6 = 3 * width6
    bar_num = 4
    for i in range(group_num):
        for b in range(bar_num):
            bb = b - bar_num//2
            if i < rad6.shape[0]:
                igap = gap6[i] / np.sin(rad6[i])
                ira6 = r_mid * np.sin(rad6[i]) + R
                irc6 = ira6 / np.tan(rad6[i])
                ic60 = irc6 - r_mid * np.cos(rad6[i]) + igap*bb
                ic61 = ic60 + width6[i] / np.tan(rad6[i])
                ip_up60 = - (irc6 - (length6 / 2) * np.cos(rad6[i]) + bb*(igap-gap6[i]*np.sin(rad6[i])))
                ip_down60 = irc6 + (length6 / 2) * np.cos(rad6[i]) + bb*(igap-gap6[i]*np.sin(rad6[i]))
                ip_up61 = ip_up60 - width6[i] / np.tan(rad6[i])
                ip_down61 = ip_down60 + width6[i] / np.tan(rad6[i])
            else:
                j = group_num - i - 1
                igap = gap6[i] / np.sin(rad6[j])
                ira6 = r_mid * np.sin(rad6[j]) + R
                irc6 = ira6 / np.tan(rad6[j])
                ic60 = - irc6 + r_mid * np.cos(rad6[j]) - igap*bb
                ic61 = ic60 - width6[i] / np.tan(rad6[j])
                ip_up60 = irc6 + (length6 / 2) * np.cos(rad6[j]) - 2*r_mid*np.cos(rad6[j]) + bb*(igap-gap6[i]*np.sin(rad6[j]))
                ip_down60 = - (irc6 - (length6 / 2) * np.cos(rad6[j]) - 2*r_mid*np.cos(rad6[j]) + bb*(igap-gap6[i]*np.sin(rad6[j])))
                ip_up61 = ip_up60 + width6[i] / np.tan(rad6[j])
                ip_down61 = ip_down60 - width6[i] / np.tan(rad6[j])
            # outer
            vprim60 = ([3, lac6, 0] +               # paraboloid, LAC, precedence index
                       [-ic60*x for x in vec] +     # center of the object
                       [ira6, ira6, irc6])          # [ra, rb, rc]
            vbasis60 = vbasis0
            vprim60 = np.hstack([vprim60, vbasis60])
            vclip60 = [ip_up60] + [-x for x in vec] + [ip_down60] + vec
            v60 = np.hstack([vprim60, vclip60])
            phantom_def_matrix = np.vstack([phantom_def_matrix, v60])
            # inner
            vprim61 = ([3, -lac6, 0] +              # paraboloid, LAC, precedence index
                       [-ic61*x for x in vec] +     # center of the object
                       [ira6, ira6, irc6])          # [ra, rb, rc]
            vbasis61 = vbasis0
            vprim61 = np.hstack([vprim61, vbasis61])
            vclip61 = [ip_up61] + [-x for x in vec] + [ip_down61] + vec
            v61 = np.hstack([vprim61, vclip61])
            phantom_def_matrix = np.vstack([phantom_def_matrix, v61])

    # ---- Object 7: aramid (nine groups of bar patterns)
    lac7 = mu_aramid - mu_rubber
    rad7 = np.deg2rad(np.array([28, 20, 12, 4]))
    group_num = 2*rad6.shape[0]
    # define size and position of each object group
    r_mid = (in_r + out_r)/2
    length7 = 8
    dlpcm = 4.0
    width7 = 5.0 / (8.0 + dlpcm*np.arange(group_num))
    gap7 = 3*width7
    bar_num = 4
    for i in range(group_num):
        for b in range(bar_num):
            bb = b - bar_num//2
            if i < rad6.shape[0]:
                igap = gap7[i] / np.sin(rad7[i])
                ira7 = r_mid * np.sin(rad7[i]) + R
                irc7 = ira7 / np.tan(rad7[i])
                ic70 = - irc7 + r_mid * np.cos(rad7[i]) - igap*bb
                ic71 = ic70 - width7[i] / np.tan(rad7[i])
                ip_up70 = irc7 + (length7/2)*np.cos(rad7[i]) + bb*(igap-gap7[i]*np.sin(rad7[i]))
                ip_down70 = -(irc7 - (length7/2)*np.cos(rad7[i]) + bb*(igap-gap7[i]*np.sin(rad7[i])))
                ip_up71 = ip_up70 + width7[i] / np.tan(rad7[i])
                ip_down71 = ip_down70 - width7[i] / np.tan(rad7[i])
            else:
                j = group_num - i - 1
                igap = gap7[i] / np.sin(rad7[j])
                ira7 = r_mid * np.sin(rad7[j]) + R
                irc7 = ira7 / np.tan(rad7[j])
                ic70 = irc7 - r_mid * np.cos(rad7[j]) + igap*bb
                ic71 = ic70 + width7[i] / np.tan(rad7[j])
                ip_up70 = - (irc7 - (length7 / 2) * np.cos(rad7[j]) - 2*r_mid*np.cos(rad7[j]) + bb*(igap-gap7[i]*np.sin(rad7[j])))
                ip_down70 = (irc7 + (length7 / 2) * np.cos(rad7[j]) - 2*r_mid*np.cos(rad7[j])) + bb*(igap-gap7[i]*np.sin(rad7[j]))
                ip_up71 = ip_up70 - width7[i] / np.tan(rad7[j])
                ip_down71 = ip_down70 + width7[i] / np.tan(rad7[j])
            # outer
            vprim70 = ([3, lac7, 0] +               # paraboloid, LAC, precedence index
                       [-ic70*x for x in vec] +     # center of the object
                       [ira7, ira7, irc7])          # [ra, rb, rc]
            vbasis70 = vbasis0
            vprim70 = np.hstack([vprim70, vbasis70])
            vclip70 = [ip_up70] + [-x for x in vec] + [ip_down70] + vec
            v70 = np.hstack([vprim70, vclip70])
            phantom_def_matrix = np.vstack([phantom_def_matrix, v70])
            # inner
            vprim71 = ([3, -lac7, 0] +              # paraboloid, LAC, precedence index
                       [-ic71*x for x in vec] +     # center of the object
                       [ira7, ira7, irc7])          # [ra, rb, rc]
            vbasis71 = vbasis0
            vprim71 = np.hstack([vprim71, vbasis71])
            vclip71 = [ip_up71] + [-x for x in vec] + [ip_down71] + vec
            v71 = np.hstack([vprim71, vclip71])
            phantom_def_matrix = np.vstack([phantom_def_matrix, v71])

    # ---- Object 8: air (grooves)
    lac8 = - mu_rubber
    group_num = 4
    # define size and position of each object group
    gap = 65
    width8 = 16
    depth8 = 12
    first_height = width/2 - (width-(group_num-1)*(gap+width8))
    for i in range(group_num):
        # outer
        vprim80 = ([2, lac8, 0] +                               # cylinder, LAC, precedence index
                   [(i*gap-first_height)*x for x in vec] +      # center of the object
                   [ra_out, ra_out, width8])                    # [ra, rb, rc]
        vbasis80 = vbasis0
        vprim80 = np.hstack([vprim80, vbasis80])
        vclip80 = [width8/2] + [-x for x in vec] + [width8/2] + vec
        v80 = np.hstack([vprim80, vclip80])
        phantom_def_matrix = np.vstack([phantom_def_matrix, v80])
        # inner
        vprim81 = ([2, -lac8, 0] +                              # cylinder, LAC, precedence index
                   [(i*gap-first_height)*x for x in vec] +      # center of the object
                   [ra_out-depth8, ra_out-depth8, width8])      # [ra, rb, rc]
        vbasis81 = vbasis0
        vprim81 = np.hstack([vprim81, vbasis81])
        vclip81 = vclip80.copy()
        v81 = np.hstack([vprim81, vclip81])
        phantom_def_matrix = np.vstack([phantom_def_matrix, v81])

    # ---- Object 9: bead
    lac9 = mu_steel - lac0
    R9 = R - T/2 + in_thickness/3
    r9 = 15
    height9 = width/2 - bar_thickness/4
    vprim90 = ([6, lac9, 0] +  # torus, LAC, precedence index
             [height9*x for x in vec] + # center of the object
             [r9, R9, 0])  # [r, R, 0]
    vbasis9 = vbasis0
    vprim90 = np.hstack([vprim90, vbasis9])
    v90 = np.hstack([vprim90, vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix, v90])

    vprim91 = ([6, lac9, 0] +  # torus, LAC, precedence index
               [-height9*x for x in vec] +  # center of the object
               [r9, R9, 0])  # [r, R, 0]
    vprim91 = np.hstack([vprim91, vbasis9])
    v91 = np.hstack([vprim91, vclip0])
    phantom_def_matrix = np.vstack([phantom_def_matrix, v91])

    return phantom_def_matrix

