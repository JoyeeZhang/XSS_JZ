import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import importlib
import os
import sys
from XSSModules import geometry as geo
from XSSModules import enum_list as enm
from XSSModules import image as img
from XSSModules import file_io as fio

'''------------------------------------------------------------------------
[F] - gamma(t)
[I] - t: array or np array
[O] - y(t) = sin(t)/t + (cos(t)-1)/t^2
[N] - Aux function for ramp filter
[H] - 20250425 created by ZYu
------------------------------------------------------------------------'''
def gamma(t):
    t = np.asarray(t)
    idx0 = np.abs(t) < np.finfo(float).eps
    idx1 = ~idx0
    y = np.empty_like(t)
    y[idx0] = 0.5
    y[idx1] = np.sin(t[idx1]) / t[idx1] + (np.cos(t[idx1]) - 1) / (t[idx1] ** 2)
    return y

'''------------------------------------------------------------------------
[F] - aux_calculate_overlap_between_ray_pixel_FB_Curve_clip (numpy version)
[I] - ttxL, ttxH ([nx,ny]): lower and upper x-coordinates of the intersection of the ray with the image pixel edges
[I] - ttyL, ttyH ([nx,ny]): lower and upper y-coordinates of the intersection of the ray with the image pixel edges
[I] - costheta, sintheta: cosine and sine of the ray angle
[I] - xxedge, yyedge ([nx+1,ny+1]): x and y coordinates of the image pixel edges
[I] - ax, ay: x and y coordinates of the source position
[I] - dxy: pixel area
[O] - sclip ([nx,ny]): reserved area of the image pixel that is clipped by the ray
[N] - Calculate the researved area of the image pixel that is clipped by the ray
[H] - 20250425 created by ZYu
------------------------------------------------------------------------'''
def aux_calculate_overlap_between_ray_pixel_FB_Curve_clip(ttxL, ttxH, ttyL, ttyH, costheta, sintheta, xxedge, yyedge, ax, ay, dxy):
    # Length of intersection
    tts = np.maximum(ttxL, ttyL)
    tte = np.minimum(ttxH, ttyH)
    dtt = np.maximum(tte - tts, 0)

    # Number of clipped vertices
    dvertex = (xxedge - ax) * costheta + (yyedge - ay) * sintheta
    dLL = dvertex[:-1, :-1]
    dLR = dvertex[:-1, 1:]
    dTL = dvertex[1:, :-1]
    dTR = dvertex[1:, 1:]

    iipos = dvertex > 0
    iiLL = iipos[:-1, :-1]
    iiLR = iipos[:-1, 1:]
    iiTL = iipos[1:, :-1]
    iiTR = iipos[1:, 1:]
    iiall = iiLL.astype(int) + iiLR.astype(int) + iiTL.astype(int) + iiTR.astype(int) # Number of vertices within the retained/clipped area

    # Clipped portion of the image
    # Start with pixel size dxy
    sclip = dxy * (iiall > 0)

    # Case 1: 1 Vertex within the clipped area
    ii1 = (iiall == 1)
    dii1 = np.zeros_like(sclip)
    dii1[ii1] = (
        np.maximum(dLL[ii1], 0) +
        np.maximum(dLR[ii1], 0) +
        np.maximum(dTL[ii1], 0) +
        np.maximum(dTR[ii1], 0)
    )
    sclip[ii1] = dii1[ii1] * dtt[ii1] * 0.5

    # Case 2: 2 Vertices within the clipped area
    ii2 = (iiall == 2)
    dii2 = np.zeros_like(sclip)
    dii2[ii2] = (
        np.maximum(dLL[ii2], 0) +
        np.maximum(dLR[ii2], 0) +
        np.maximum(dTL[ii2], 0) +
        np.maximum(dTR[ii2], 0)
    )
    sclip[ii2] = dii2[ii2] * dtt[ii2] * 0.5

    # Case 3: 3 Vertices within the clipped area
    ii3 = (iiall == 3)
    dii3 = np.zeros_like(sclip)
    dii3[ii3] = (
        np.minimum(dLL[ii3], 0) +
        np.minimum(dLR[ii3], 0) +
        np.minimum(dTL[ii3], 0) +
        np.minimum(dTR[ii3], 0)
    )
    sclip[ii3] = dxy + dii3[ii3] * dtt[ii3] * 0.5

    return sclip

'''------------------------------------------------------------------------
[F] - aux_calculate_overlap_between_ray_pixel_FB_Curve_clip (cupy version)
[I] - ttxL, ttxH ([nx,ny]): lower and upper x-coordinates of the intersection of the ray with the image pixel edges
[I] - ttyL, ttyH ([nx,ny]): lower and upper y-coordinates of the intersection of the ray with the image pixel edges
[I] - costheta, sintheta: cosine and sine of the ray angle
[I] - xxedge, yyedge ([nx+1,ny+1]): x and y coordinates of the image pixel edges
[I] - ax, ay: x and y coordinates of the source position
[I] - dxy: pixel area
[O] - sclip ([nx,ny]): reserved area of the image pixel that is clipped by the ray
[N] - Calculate the researved area of the image pixel that is clipped by the ray
[H] - 20250425 created by ZYu
------------------------------------------------------------------------'''
def aux_calculate_overlap_between_ray_pixel_FB_Curve_clip_GPU(ttxL, ttxH, ttyL, ttyH, costheta, sintheta, xxedge, yyedge, ax, ay, dxy):
    # Length of intersection
    tts = cp.maximum(ttxL, ttyL)
    tte = cp.minimum(ttxH, ttyH)
    dtt = cp.maximum(tte - tts, 0)

    # Number of clipped vertices
    dvertex = (xxedge - ax) * costheta + (yyedge - ay) * sintheta
    dLL = dvertex[:-1, :-1]
    dLR = dvertex[:-1, 1:]
    dTL = dvertex[1:, :-1]
    dTR = dvertex[1:, 1:]

    iipos = dvertex > 0
    iiLL = iipos[:-1, :-1]
    iiLR = iipos[:-1, 1:]
    iiTL = iipos[1:, :-1]
    iiTR = iipos[1:, 1:]
    iiall = iiLL.astype(int) + iiLR.astype(int) + iiTL.astype(int) + iiTR.astype(int) # Number of vertices within the retained/clipped area

    # Clipped portion of the image
    # Start with pixel size dxy
    sclip = dxy * (iiall > 0)

    # Case 1: 1 Vertex within the clipped area
    ii1 = (iiall == 1)
    dii1 = cp.zeros_like(sclip)
    dii1[ii1] = (
        cp.maximum(dLL[ii1], 0) +
        cp.maximum(dLR[ii1], 0) +
        cp.maximum(dTL[ii1], 0) +
        cp.maximum(dTR[ii1], 0)
    )
    sclip[ii1] = dii1[ii1] * dtt[ii1] * 0.5

    # Case 2: 2 Vertices within the clipped area
    ii2 = (iiall == 2)
    dii2 = cp.zeros_like(sclip)
    dii2[ii2] = (
        cp.maximum(dLL[ii2], 0) +
        cp.maximum(dLR[ii2], 0) +
        cp.maximum(dTL[ii2], 0) +
        cp.maximum(dTR[ii2], 0)
    )
    sclip[ii2] = dii2[ii2] * dtt[ii2] * 0.5

    # Case 3: 3 Vertices within the clipped area
    ii3 = (iiall == 3)
    dii3 = cp.zeros_like(sclip)
    dii3[ii3] = (
        cp.minimum(dLL[ii3], 0) +
        cp.minimum(dLR[ii3], 0) +
        cp.minimum(dTL[ii3], 0) +
        cp.minimum(dTR[ii3], 0)
    )
    sclip[ii3] = dxy + dii3[ii3] * dtt[ii3] * 0.5

    return sclip

'''------------------------------------------------------------------------
[F] - apply rowwise ramp filter for flat geometry: [nv,nu] or [nlambda, nu]
[I] - g0: fanogram, np.array of size [nlambda, nu] or [nv,nu]
[I] - alpha: filter parameter (0.5 for Shepp-Logan, 1.0 for Ram-Lak)
[I] - ds: channel spacing
[O] - I1: filtered image of the same size as I0
[H] - 20250425 created by ZYu
------------------------------------------------------------------------'''
def apply_ramp_filter_rowwise_flat(g0, alpha, ds):
    nrow, ncol = g0.shape
    N = 2 ** int(np.ceil(np.log2(ncol)))

    # Construct ramp filter kernel
    tmp = np.pi * np.arange(-ncol + 1, ncol)
    hbramp = alpha * gamma(tmp) + 0.5 * (1 - alpha) * (gamma(tmp + np.pi) + gamma(tmp - np.pi))
    hbramp = 0.5 * hbramp / ds

    # Zero padding the filter to length 2N
    zp1 = np.zeros(2 * (N - ncol) + 1)
    hbramp_full = np.concatenate([hbramp[ncol-1:2*ncol-1], zp1, hbramp[:ncol-1]])
    Hbramp = np.fft.fft(hbramp_full)

    # Zero padding the input image along the rows (columns in transposed version)
    zp2 = np.zeros((2 * N - ncol, nrow))
    g1 = np.vstack((g0.T, zp2))

    # Apply ramp filter in frequency domain
    HMramp = Hbramp[:, np.newaxis]  # Make it 2D for broadcasting
    G1 = np.fft.fft(g1, axis=0)
    GF = G1 * HMramp
    gF = np.fft.ifft(GF, axis=0).real[:ncol, :].T

    return gF

'''------------------------------------------------------------------------
[F] - perform_BP_FB_Flat
[I] - gF ([nlambda,nu]): filtered fanogram
[I] - slice: img.Slice2D object containing image slice information and data
[I] - uoff,du: detector channel grid definition
        uk = (k - uoff) * du, where k is the channel index 0,1,..., nu-1
[I] - lambdai,dlambda: view angle grid definition, lambdak = lambdai + k * dlambda
[I] - R: source to isocenter distance
[I] - D: source to detector distance
[I] - w [optional]: redundancy weighting, default is 0.5
[O] - slice: updated slice.slice
[H] - 20250425 created by ZYu
------------------------------------------------------------------------'''
def perform_BP_FB_Flat(gF,slice:img.Slice2D,uoff,du,lambdai,dlambda,R,D,w = None):
    # If not specified, set redundancy weighting w to 1/2
    if w is None:
        w = 0.5
        
    nlambda,nu = gF.shape
    u_lowerbound = -uoff * du
    u_upperbound = (nu - uoff - 1) * du

    # XY grids
    xoff = slice.ocol
    dx = slice.dcol
    nx = slice.ncol
    yoff = slice.orow
    dy = slice.drow
    ny = slice.nrow
    x = -xoff*dx + np.arange(nx) * dx
    y = -yoff*dy + np.arange(ny) * dy
    yy, xx = np.meshgrid(y, x, indexing='ij')  # shape: (nx, ny)

    # Rotational angles
    lambdas = lambdai + np.arange(nlambda) * dlambda
    cosla = np.cos(lambdas)
    sinla = np.sin(lambdas)

    # Initialize image
    im = np.zeros((ny, nx))
    fig = plt.figure()
    ax = fig.add_subplot()

    for k in range(nlambda):
        sin_k = sinla[k]
        cos_k = cosla[k]

        denom = R - xx * cos_k - yy * sin_k
        uu = D * (-xx * sin_k + yy * cos_k) / denom

        # Create mask for valid uu
        valid_mask = (uu > u_lowerbound) & (uu < u_upperbound)

        # Process only valid pixels
        if not np.any(valid_mask):
            continue

        uu_valid = uu[valid_mask]
        uk = uu_valid / du + uoff
        uk_lb = np.floor(uk).astype(int)
        uk_ub = uk_lb + 1
        uk_ld = uk - uk_lb
        uk_ud = 1 - uk_ld

        # Extract fano column
        gFk = gF[k, :]
        gFk_interp = gFk[uk_lb] * uk_ud + gFk[uk_ub] * uk_ld

        # Interpolation coefficients
        coefk = 1.0 / (denom[valid_mask] ** 2)

        # Update image
        im[valid_mask] += gFk_interp * coefk 

        if k % 50 == 0:
            ttl = f'Completed projection #: {k}'
            print(ttl)
            ax.clear()
            ax.set_title(ttl)
            plt.imshow(im, cmap='gray')
            plt.axis('image')
            plt.axis('off')
            plt.pause(0.01)
            

    im = w*R*D*abs(dlambda)*im
    slice.slice = im
    
'''------------------------------------------------------------------------
[F] - perform_FP_FB_Curve_Joseph
[I] - slice: img.Slice2D object containing image slice information and data
[I] - lambdas,dlambda,nlambda: view angle grid definition, lambdak = lambdas + k * dlambda
[I] - gammas,dgamma,ngamma: fan angle grid definition, gammak = gammas + k * dgamma
[I] - R: source to isocenter distance
[O] - g: fanogram of size [nlambda, ngamma]
[N] - Perform forward projection for fan-beam geometry (curved geo) with Joseph's method
[H] - 20250425 created by ZYu
------------------------------------------------------------------------'''
def perform_FP_FB_Curve_Joseph(slice:img.Slice2D,lambdas,dlambda,nlambda,gammas, dgamma, ngamma,R):
    xoff = slice.ocol
    dx = slice.dcol
    nx = slice.ncol
    yoff = slice.orow
    dy = slice.drow
    ny = slice.nrow
    
    x = -xoff*dx + np.arange(nx) * dx
    y = -yoff*dy + np.arange(ny) * dy
    
    lamb = lambdas + np.arange(nlambda) * dlambda
    gamma = gammas + np.arange(ngamma) * dgamma
    s = R * np.sin(gamma)
    
    im = slice.slice

    g = np.zeros((nlambda,ngamma))
    tedge = np.sqrt(2) / 2.0
    
    for kla in range(nlambda):
        theta = np.pi / 2.0 + lamb[kla] - gamma

        # Beam closer to x-axis: np.where returns a tuple of arrays for multidimensional inputs, there for [0]
        iix = np.where(np.abs(np.sin(theta)) >= tedge)[0]
        niix = len(iix)
        if iix.size > 0:
            tmpx = np.zeros((niix, nx))
            sx = np.outer(s[iix], np.ones(nx))
            xx = np.outer(np.ones(niix), x)
            cosx = np.outer(np.cos(theta[iix]), np.ones(nx))
            sinx = np.outer(np.sin(theta[iix]), np.ones(nx))
            ixx = np.outer(np.ones(niix), np.arange(nx))

            ystar = (sx - xx * cosx) / sinx # vx * vtheta = s
            ky = ystar/dy + yoff
            ky = ky.flatten()
            ii = np.where((ky >= 0) & (ky < ny-1))[0]
            ky_lb = np.floor(ky[ii]).astype(int)
            ky_ld = ky[ii] - ky_lb
            xx_ii = ixx.flatten()[ii].astype(int)

            tmpx_flat = np.zeros_like(ky_ld)
            im_flat = im.flatten()

            ky_idx1 = ky_lb * nx + xx_ii
            ky_idx2 = (ky_lb+1) * nx + xx_ii

            tmpx_flat = im_flat[ky_idx1] * (1 - ky_ld) + im_flat[ky_idx2] * ky_ld
            tmpx[np.unravel_index(ii, tmpx.shape)] = tmpx_flat # put tmpx_lat into the ii index of tmpx
            gk = np.sum(tmpx, axis=1) * dx / np.abs(sinx[:, 0])
            g[kla, iix] = gk
        # Beam closer to y-axis
        iiy = np.where(np.abs(np.sin(theta)) < tedge)[0]
        if iiy.size > 0:
            tmpy = np.zeros((len(iiy), ny))
            sy = np.outer(s[iiy], np.ones(ny))
            yy = np.outer(np.ones(len(iiy)), y)
            cosy = np.outer(np.cos(theta[iiy]), np.ones(ny))
            siny = np.outer(np.sin(theta[iiy]), np.ones(ny))
            iyy = np.outer(np.ones(len(iiy)), np.arange(ny))

            xstar = (sy - yy * siny) / cosy
            kx = xstar / dx + xoff
            kx = kx.flatten()
            jj = np.where((kx >= 0) & (kx < nx-1))[0]
            kx_lb = np.floor(kx[jj]).astype(int)
            kx_ld = kx[jj] - kx_lb
            yy_jj = iyy.flatten()[jj].astype(int)

            im_flat = im.flatten()
            kx_idx1 = yy_jj* nx + kx_lb
            kx_idx2 = kx_idx1 + 1

            tmpy_flat = im_flat[kx_idx1] * (1 - kx_ld) + im_flat[kx_idx2] * kx_ld
            tmpy[np.unravel_index(jj, tmpy.shape)] = tmpy_flat
            
            g[kla,iiy] = np.sum(tmpy, axis=1) * dy / np.abs(cosy[:, 0])

    return g

'''------------------------------------------------------------------------
[F] - perform_FP_FB_Flat_Joseph
[I] - slice: img.Slice2D object containing image slice information and data
[I] - lambdas,dlambda,nlambda: view angle grid definition, lambdak = lambdas + k * dlambda
[I] - uoff,du,nu: detector channel grid definition, uk = (k - uoff) * du
[I] - R: source to isocenter distance
[I] - D: source to detector distance
[I] - nups_beam_avg: number of upsampling for each beamlet, default is 1
      Up sampling is only allowed with odd number of nups_beam_avg
      The middle of the upsampled beamlet is aligned with the original beamlet
[O] - g: fanogram of size [nlambda, nu]
[N] - Perform forward projection for fan-beam geometry (flat geo) with Joseph's method
[H] - 20250425 created by ZYu
------------------------------------------------------------------------'''    
def perform_FP_FB_Flat_Joseph(slice:img.Slice2D, lambdas, dlambda, nlambda, uoff, du, nu, R, D, nups_beam_avg=1):
    nu_original = nu
    
    nu = nups_beam_avg*nu
    du = du/nups_beam_avg
    uoff = uoff*nups_beam_avg + 0.5*(nups_beam_avg-1)
    
    xoff = slice.ocol
    dx = slice.dcol
    nx = slice.ncol
    yoff = slice.orow
    dy = slice.drow
    ny = slice.nrow
    
    x = -xoff*dx + np.arange(nx) * dx
    y = -yoff*dy + np.arange(ny) * dy
    
    lamb = lambdas + np.arange(nlambda) * dlambda
    u = (np.arange(nu) - uoff) * du
    gamma = np.atan(u/D) # atan returns radian angle between -pi/2 and pi/2
    s = R * np.sin(gamma)
    
    im = slice.slice

    g = np.zeros((nlambda,nu))
    tedge = np.sqrt(2) / 2.0
    
    for kla in range(nlambda):
        theta = np.pi / 2.0 + lamb[kla] - gamma

        # Beam closer to x-axis: np.where returns a tuple of arrays for multidimensional inputs, there for [0]
        iix = np.where(np.abs(np.sin(theta)) >= tedge)[0]
        niix = len(iix)
        if iix.size > 0:
            tmpx = np.zeros((niix, nx))
            sx = np.outer(s[iix], np.ones(nx))
            xx = np.outer(np.ones(niix), x)
            cosx = np.outer(np.cos(theta[iix]), np.ones(nx))
            sinx = np.outer(np.sin(theta[iix]), np.ones(nx))
            ixx = np.outer(np.ones(niix), np.arange(nx))

            ystar = (sx - xx * cosx) / sinx # vx * vtheta = s
            ky = ystar/dy + yoff
            ky = ky.flatten()
            ii = np.where((ky >= 0) & (ky < ny-1))[0]
            ky_lb = np.floor(ky[ii]).astype(int)
            ky_ld = ky[ii] - ky_lb
            xx_ii = ixx.flatten()[ii].astype(int)

            tmpx_flat = np.zeros_like(ky_ld)
            im_flat = im.flatten()

            ky_idx1 = ky_lb * nx + xx_ii
            ky_idx2 = (ky_lb+1) * nx + xx_ii

            tmpx_flat = im_flat[ky_idx1] * (1 - ky_ld) + im_flat[ky_idx2] * ky_ld
            tmpx[np.unravel_index(ii, tmpx.shape)] = tmpx_flat # put tmpx_lat into the ii index of tmpx
            gk = np.sum(tmpx, axis=1) * dx / np.abs(sinx[:, 0])
            g[kla, iix] = gk
        # Beam closer to y-axis
        iiy = np.where(np.abs(np.sin(theta)) < tedge)[0]
        if iiy.size > 0:
            tmpy = np.zeros((len(iiy), ny))
            sy = np.outer(s[iiy], np.ones(ny))
            yy = np.outer(np.ones(len(iiy)), y)
            cosy = np.outer(np.cos(theta[iiy]), np.ones(ny))
            siny = np.outer(np.sin(theta[iiy]), np.ones(ny))
            iyy = np.outer(np.ones(len(iiy)), np.arange(ny))

            xstar = (sy - yy * siny) / cosy
            kx = xstar / dx + xoff
            kx = kx.flatten()
            jj = np.where((kx >= 0) & (kx < nx-1))[0]
            kx_lb = np.floor(kx[jj]).astype(int)
            kx_ld = kx[jj] - kx_lb
            yy_jj = iyy.flatten()[jj].astype(int)

            im_flat = im.flatten()
            kx_idx1 = yy_jj* nx + kx_lb
            kx_idx2 = kx_idx1 + 1

            tmpy_flat = im_flat[kx_idx1] * (1 - kx_ld) + im_flat[kx_idx2] * kx_ld
            tmpy[np.unravel_index(jj, tmpy.shape)] = tmpy_flat
            
            g[kla,iiy] = np.sum(tmpy, axis=1) * dy / np.abs(cosy[:, 0])
    
    # Downsample by averaging
    g = g.reshape(nlambda,1,nu_original,nups_beam_avg).mean(axis=(1,3))

    return g    

'''------------------------------------------------------------------------
[F] - perform_FP_FB_Curve_Fidelity
[I] - slice: img.Slice2D object containing image slice information and data
[I] - lambdas,dlambda,nlambda: view angle grid definition, lambdak = lambdas + k * dlambda
[I] - gammas,dgamma,ngamma: fan angle grid definition, gammak = gammas + k * dgamma
[I] - R: source to isocenter distance
[O] - g: fanogram of size [nlambda, ngamma]
[N] - Perform forward projection for fan-beam geometry (curved geo) with High Fidelity method
[H] - 20250425 created by ZYu
------------------------------------------------------------------------'''
def perform_FP_FB_Curve_Fidelity(slice:img.Slice2D,lambdas,dlambda,nlambda,gammas, dgamma, ngamma,R):    
    # Initialize fanogram
    g = np.zeros((nlambda, ngamma))
    im = slice.slice
    
    # Slice grids (pixel edges)
    xoff = slice.ocol
    dx = slice.dcol
    nx = slice.ncol
    yoff = slice.orow
    dy = slice.drow
    ny = slice.nrow
    dxy = dx * dy # pixel area
    
    x = -xoff*dx + np.arange(nx) * dx
    y = -yoff*dy + np.arange(ny) * dy    
    xedge = np.append(x - 0.5 * dx, x[-1] + 0.5 * dx) # 1D array: nx+1
    yedge = np.append(y - 0.5 * dy, y[-1] + 0.5 * dy) # 1D array: ny+1
    xx, yy = np.meshgrid(x, y)# [ny,nx]
    xxedge, yyedge = np.meshgrid(xedge, yedge) # [ny+1,nx+1]
    
    # View angles (center)
    lamb = lambdas + np.arange(nlambda) * dlambda

    # Fan angles (beam edges)
    gammas = gammas - 0.5 * dgamma
    ngamma = ngamma + 1
    gamma = gammas + np.arange(ngamma) * dgamma

    # Forward projection
    for kla in range(nlambda):
        lambdak = lamb[kla] # scalar

        # vtheta: clipping line normal
        theta = np.pi/2 + lambdak - gamma # 1D array: ngamma
        # vthetaper: equvalent to valpha
        thetaper = theta + np.pi/2 # 1D array: ngamma

        sintheta = np.sin(theta) # 1D array: ngamma
        costheta = np.cos(theta)
        sinthetaper = np.sin(thetaper)
        costhetaper = np.cos(thetaper)

        # Source position
        ax = R * np.cos(lambdak) # scalar
        ay = R * np.sin(lambdak) # scalar

        # Edge distances
        txedge = xedge - ax # 1D array: nx+1
        tyedge = yedge - ay # 1D array: ny+1
        
        iixparallel = np.where(np.diff(np.sign(txedge))>0)[0]
        iiyparallel = np.where(np.diff(np.sign(tyedge))>0)[0]

        txL = np.zeros((nx, ngamma))
        txH = np.zeros((nx, ngamma))
        tyL = np.zeros((ny, ngamma))
        tyH = np.zeros((ny, ngamma))
        
        # X-Direction
        iiX1 = np.abs(costhetaper)>np.finfo(float).eps
        iiX0 = ~iiX1 # when ray is parallel to the y-axis
        
        txedge1 = txedge[:,None]/costhetaper[iiX1]
        txL[:,iiX1] = np.fmin(txedge1[:-1,:],txedge1[1:,:])
        txH[:,iiX1] = np.fmax(txedge1[:-1,:],txedge1[1:,:])
        txL[:,iiX0] = np.finfo(float).max
        txH[:,iiX0] = 0
        
        if iixparallel.size > 0:
            txL[iixparallel[-1],iiX0] = 0
            txH[iixparallel[-1],iiX0] = np.finfo(float).max
        
        # Y-Direction
        iiY1 = np.abs(sinthetaper)>np.finfo(float).eps
        iiY0 = ~iiY1 # when ray is parallel to the x-axis
        
        tyedge1 = tyedge[:,None] / sinthetaper[iiY1]
        tyL[:,iiY1] = np.fmin(tyedge1[:-1,:],tyedge1[1:,:])
        tyH[:,iiY1] = np.fmax(tyedge1[:-1,:],tyedge1[1:,:])
        tyL[:,iiY0] = np.finfo(float).max
        tyH[:,iiY0] = 0
        if iiyparallel.size > 0:
            tyL[iiyparallel[-1],iiY0] = 0 # [-1] to include the case when the ray overlaps with the y-axis
            tyH[iiyparallel[-1],iiY0] = np.finfo(float).max
        
        # First projection
        kgamma = 0
        ttxL = np.tile(txL[:, kgamma][None,:], (ny,1)) # [ny,nx]
        ttxH = np.tile(txH[:, kgamma][None,:], (ny,1)) # [ny,nx]
        ttyL = np.tile(tyL[:, kgamma][:,None], (1,nx)) # [ny,nx]
        ttyH = np.tile(tyH[:, kgamma][:,None], (1,nx)) # [ny,nx]

        # Calculate the overlap between each beam and each pixel for a single view: 
        # ttxL, ttxH, ttyL, ttyH: [nx,ny]
        # xxedge, yyedge: [nx+1,ny+1]
        # sclip: [nx,ny]
        sclipL = aux_calculate_overlap_between_ray_pixel_FB_Curve_clip(ttxL, ttxH, ttyL, ttyH, costheta[kgamma], sintheta[kgamma], xxedge, yyedge, ax, ay, dxy)
        
        # fig = plt.figure()
        # axx = fig.add_subplot()

        for kgamma in range(1, ngamma):
            # if kgamma == 139 and kla == 5:
            #     print('debugging')
            ttxL = np.tile(txL[:, kgamma][None,:], (ny,1)) # [ny,nx]
            ttxH = np.tile(txH[:, kgamma][None,:], (ny,1)) # [ny,nx]
            ttyL = np.tile(tyL[:, kgamma][:,None], (1,nx)) # [ny,nx]
            ttyH = np.tile(tyH[:, kgamma][:,None], (1,nx)) # [ny,nx]
            # dsclip = overlap corresponding to the lower beam edge - overlap of corresponding to the upper beam edge
            sclipH = aux_calculate_overlap_between_ray_pixel_FB_Curve_clip(ttxL, ttxH, ttyL, ttyH, costheta[kgamma], sintheta[kgamma], xxedge, yyedge, ax, ay, dxy)
            dsclip = sclipL - sclipH
            
            # axx.clear()
            # axx.set_title(f'{kla} - {kgamma}')
            # plt.imshow(dsclip, cmap='gray')
            # plt.axis('image')
            # plt.axis('off')
            # plt.pause(0.01)

            # Contribution only from intersected regions
            iik = dsclip > 0
            spd = np.sqrt((ax - xx[iik])**2 + (ay - yy[iik])**2) # weighted by source to pixel center distance
            g[kla,kgamma - 1] = np.sum(im[iik] * dsclip[iik] / spd) 

            sclipL = sclipH

        if kla % 10 == 0:
            print(f'Completed view #{kla + 1}')
            
    g = g/dgamma

    return g

'''------------------------------------------------------------------------
[F] - perform_FP_FB_Flat_Fidelity [numpy version]
[I] - slice: img.Slice2D object containing image slice information and data
[I] - lambdas,dlambda,nlambda: view angle grid definition, lambdak = lambdas + k * dlambda
[I] - uoff,du,nu: detector channel grid definition, uk = (k - uoff) * du
[I] - R: source to isocenter distance
[I] - D: source to detector distance
[O] - g: fanogram of size [nlambda, nu]
[N] - Perform forward projection for fan-beam geometry (flat geo) with High Fidelity method
[H] - 20250425 created by ZYu
------------------------------------------------------------------------'''
def perform_FP_FB_Flat_Fidelity(slice:img.Slice2D,lambdas,dlambda,nlambda,uoff,du,nu,R,D):    
    # Initialize fanogram
    g = np.zeros((nlambda, nu))
    im = slice.slice
    
    # Slice grids (pixel edges)
    xoff = slice.ocol
    dx = slice.dcol
    nx = slice.ncol
    yoff = slice.orow
    dy = slice.drow
    ny = slice.nrow
    dxy = dx * dy # pixel area
    
    x = -xoff*dx + np.arange(nx) * dx
    y = -yoff*dy + np.arange(ny) * dy    
    xedge = np.append(x - 0.5 * dx, x[-1] + 0.5 * dx) # 1D array: nx+1
    yedge = np.append(y - 0.5 * dy, y[-1] + 0.5 * dy) # 1D array: ny+1
    xx, yy = np.meshgrid(x, y)# [ny,nx]
    xxedge, yyedge = np.meshgrid(xedge, yedge) # [ny+1,nx+1]
    
    # View angles (center)
    lamb = lambdas + np.arange(nlambda) * dlambda

    # Channel positions corresponding to edges of each beamlet
    uoff = uoff + np.float64(0.5)
    nu = nu + 1
    u = (np.arange(nu) - uoff) * du # 1D array: nu
    gamma = np.arctan(u/D) # atan returns radian angle between -pi/2 and pi/2
    dgamma = np.diff(gamma) # 1D array: nu-1, shift variant fan angle correspnoding to each beamlet

    # Forward projection
    for kla in range(nlambda):
        lambdak = lamb[kla] # scalar

        # vtheta: clipping line normal
        theta = np.pi/2 + lambdak - gamma # 1D array: ngamma
        # vthetaper: equvalent to valpha
        thetaper = theta + np.pi/2 # 1D array: ngamma

        sintheta = np.sin(theta) # 1D array: ngamma
        costheta = np.cos(theta)
        sinthetaper = np.sin(thetaper)
        costhetaper = np.cos(thetaper)

        # Source position
        ax = R * np.cos(lambdak) # scalar
        ay = R * np.sin(lambdak) # scalar

        # Edge distances
        txedge = xedge - ax # 1D array: nx+1
        tyedge = yedge - ay # 1D array: ny+1
        
        iixparallel = np.where(np.diff(np.sign(txedge))>0)[0]
        iiyparallel = np.where(np.diff(np.sign(tyedge))>0)[0]

        txL = np.zeros((nx, nu))
        txH = np.zeros((nx, nu))
        tyL = np.zeros((ny, nu))
        tyH = np.zeros((ny, nu))
        
        # X-Direction
        iiX1 = np.abs(costhetaper)>np.finfo(float).eps
        iiX0 = ~iiX1 # when ray is parallel to the y-axis
        
        txedge1 = txedge[:,None]/costhetaper[iiX1]
        txL[:,iiX1] = np.fmin(txedge1[:-1,:],txedge1[1:,:])
        txH[:,iiX1] = np.fmax(txedge1[:-1,:],txedge1[1:,:])
        txL[:,iiX0] = np.finfo(float).max
        txH[:,iiX0] = 0
        
        if iixparallel.size > 0:
            txL[iixparallel[-1],iiX0] = 0
            txH[iixparallel[-1],iiX0] = np.finfo(float).max
        
        # Y-Direction
        iiY1 = np.abs(sinthetaper)>np.finfo(float).eps
        iiY0 = ~iiY1 # when ray is parallel to the x-axis
        
        tyedge1 = tyedge[:,None] / sinthetaper[iiY1]
        tyL[:,iiY1] = np.fmin(tyedge1[:-1,:],tyedge1[1:,:])
        tyH[:,iiY1] = np.fmax(tyedge1[:-1,:],tyedge1[1:,:])
        tyL[:,iiY0] = np.finfo(float).max
        tyH[:,iiY0] = 0
        if iiyparallel.size > 0:
            tyL[iiyparallel[-1],iiY0] = 0 # [-1] to include the case when the ray overlaps with the y-axis
            tyH[iiyparallel[-1],iiY0] = np.finfo(float).max
        
        # First projection
        kgamma = 0
        ttxL = np.tile(txL[:, kgamma][None,:], (ny,1)) # [ny,nx]
        ttxH = np.tile(txH[:, kgamma][None,:], (ny,1)) # [ny,nx]
        ttyL = np.tile(tyL[:, kgamma][:,None], (1,nx)) # [ny,nx]
        ttyH = np.tile(tyH[:, kgamma][:,None], (1,nx)) # [ny,nx]

        # Calculate the overlap between each beam and each pixel for a single view: 
        # ttxL, ttxH, ttyL, ttyH: [nx,ny]
        # xxedge, yyedge: [nx+1,ny+1]
        # sclip: [nx,ny]
        sclipL = aux_calculate_overlap_between_ray_pixel_FB_Curve_clip(ttxL, ttxH, ttyL, ttyH, costheta[kgamma], sintheta[kgamma], xxedge, yyedge, ax, ay, dxy)
        
        # fig = plt.figure()
        # axx = fig.add_subplot()

        for kgamma in range(1, nu):
            # if kgamma == 139 and kla == 5:
            #     print('debugging')
            ttxL = np.tile(txL[:, kgamma][None,:], (ny,1)) # [ny,nx]
            ttxH = np.tile(txH[:, kgamma][None,:], (ny,1)) # [ny,nx]
            ttyL = np.tile(tyL[:, kgamma][:,None], (1,nx)) # [ny,nx]
            ttyH = np.tile(tyH[:, kgamma][:,None], (1,nx)) # [ny,nx]
            # dsclip = overlap corresponding to the lower beam edge - overlap of corresponding to the upper beam edge
            sclipH = aux_calculate_overlap_between_ray_pixel_FB_Curve_clip(ttxL, ttxH, ttyL, ttyH, costheta[kgamma], sintheta[kgamma], xxedge, yyedge, ax, ay, dxy)
            dsclip = sclipL - sclipH
            
            # axx.clear()
            # axx.set_title(f'{kla} - {kgamma}')
            # plt.imshow(dsclip, cmap='gray')
            # plt.axis('image')
            # plt.axis('off')
            # plt.pause(0.01)

            # Contribution only from intersected regions
            iik = dsclip > 0
            spd = np.sqrt((ax - xx[iik])**2 + (ay - yy[iik])**2) # weighted by source to pixel center distance
            g[kla,kgamma - 1] = np.sum(im[iik] * dsclip[iik] / spd) / dgamma[kgamma - 1] # divide by the beamlet fan angle width, channel variant

            sclipL = sclipH

        if kla % 10 == 0:
            print(f'Completed view #{kla + 1}')

    return g

'''------------------------------------------------------------------------
[F] - perform_FP_FB_Flat_Fidelity [cupy version]
[I] - slice: img.Slice2D object containing image slice information and data
[I] - lambdas,dlambda,nlambda: view angle grid definition, lambdak = lambdas + k * dlambda
[I] - uoff,du,nu: detector channel grid definition, uk = (k - uoff) * du
[I] - R: source to isocenter distance
[I] - D: source to detector distance
[O] - g: fanogram of size [nlambda, nu]
[N] - Perform forward projection for fan-beam geometry (flat geo) with High Fidelity method
[H] - 20250425 created by ZYu
------------------------------------------------------------------------'''
def perform_FP_FB_Flat_Fidelity_GPU(slice:img.Slice2D,lambdas,dlambda,nlambda,uoff,du,nu,R,D):
    if cp.cuda.runtime.getDeviceCount() > 0:
        print("CUDA is working.")

    # Initialize fanogram
    g = cp.zeros((nlambda, nu))
    im = cp.asarray(slice.slice)
    
    # Slice grids (pixel edges)
    xoff = slice.ocol
    dx = slice.dcol
    nx = slice.ncol
    yoff = slice.orow
    dy = slice.drow
    ny = slice.nrow
    dxy = dx * dy # pixel area
    
    x = -xoff*dx + cp.arange(nx) * dx
    y = -yoff*dy + cp.arange(ny) * dy    
    xedge = cp.append(x - 0.5 * dx, x[-1] + 0.5 * dx) # 1D array: nx+1
    yedge = cp.append(y - 0.5 * dy, y[-1] + 0.5 * dy) # 1D array: ny+1
    xx, yy = cp.meshgrid(x, y)# [ny,nx]
    xxedge, yyedge = cp.meshgrid(xedge, yedge) # [ny+1,nx+1]
    
    # View angles (center)
    lamb = lambdas + cp.arange(nlambda) * dlambda

    # Channel positions corresponding to edges of each beamlet
    uoff = uoff + cp.float64(0.5)
    nu = nu + 1
    u = (cp.arange(nu) - uoff) * du # 1D array: nu
    gamma = cp.arctan(u/D) # atan returns radian angle between -pi/2 and pi/2
    dgamma = cp.diff(gamma) # 1D array: nu-1, shift variant fan angle correspnoding to each beamlet

    # Forward projection
    for kla in range(nlambda):
        lambdak = lamb[kla] # scalar

        # vtheta: clipping line normal
        theta = cp.pi/2 + lambdak - gamma # 1D array: ngamma
        # vthetaper: equvalent to valpha
        thetaper = theta + np.pi/2 # 1D array: ngamma

        sintheta = cp.sin(theta) # 1D array: ngamma
        costheta = cp.cos(theta)
        sinthetaper = cp.sin(thetaper)
        costhetaper = cp.cos(thetaper)

        # Source position
        ax = R * cp.cos(lambdak) # scalar
        ay = R * cp.sin(lambdak) # scalar

        # Edge distances
        txedge = xedge - ax # 1D array: nx+1
        tyedge = yedge - ay # 1D array: ny+1
        
        iixparallel = cp.where(cp.diff(cp.sign(txedge))>0)[0]
        iiyparallel = cp.where(cp.diff(cp.sign(tyedge))>0)[0]

        txL = cp.zeros((nx, nu))
        txH = cp.zeros((nx, nu))
        tyL = cp.zeros((ny, nu))
        tyH = cp.zeros((ny, nu))
        
        # X-Direction
        iiX1 = cp.abs(costhetaper)>cp.finfo(float).eps
        iiX0 = ~iiX1 # when ray is parallel to the y-axis
        
        txedge1 = txedge[:,None]/costhetaper[iiX1]
        txL[:,iiX1] = cp.fmin(txedge1[:-1,:],txedge1[1:,:])
        txH[:,iiX1] = cp.fmax(txedge1[:-1,:],txedge1[1:,:])
        txL[:,iiX0] = cp.finfo(float).max
        txH[:,iiX0] = 0
        
        if iixparallel.size > 0:
            txL[iixparallel[-1],iiX0] = 0
            txH[iixparallel[-1],iiX0] = cp.finfo(float).max
        
        # Y-Direction
        iiY1 = cp.abs(sinthetaper)>cp.finfo(float).eps
        iiY0 = ~iiY1 # when ray is parallel to the x-axis
        
        tyedge1 = tyedge[:,None] / sinthetaper[iiY1]
        tyL[:,iiY1] = cp.fmin(tyedge1[:-1,:],tyedge1[1:,:])
        tyH[:,iiY1] = cp.fmax(tyedge1[:-1,:],tyedge1[1:,:])
        tyL[:,iiY0] = cp.finfo(float).max
        tyH[:,iiY0] = 0
        if iiyparallel.size > 0:
            tyL[iiyparallel[-1],iiY0] = 0 # [-1] to include the case when the ray overlaps with the y-axis
            tyH[iiyparallel[-1],iiY0] = cp.finfo(float).max
        
        # First projection
        kgamma = 0
        ttxL = cp.tile(txL[:, kgamma][None,:], (ny,1)) # [ny,nx]
        ttxH = cp.tile(txH[:, kgamma][None,:], (ny,1)) # [ny,nx]
        ttyL = cp.tile(tyL[:, kgamma][:,None], (1,nx)) # [ny,nx]
        ttyH = cp.tile(tyH[:, kgamma][:,None], (1,nx)) # [ny,nx]

        # Calculate the overlap between each beam and each pixel for a single view: 
        # ttxL, ttxH, ttyL, ttyH: [nx,ny]
        # xxedge, yyedge: [nx+1,ny+1]
        # sclip: [nx,ny]
        sclipL = aux_calculate_overlap_between_ray_pixel_FB_Curve_clip(ttxL, ttxH, ttyL, ttyH, costheta[kgamma], sintheta[kgamma], xxedge, yyedge, ax, ay, dxy)
        
        # fig = plt.figure()
        # axx = fig.add_subplot()

        for kgamma in range(1, nu):
            # if kgamma == 139 and kla == 5:
            #     print('debugging')
            ttxL = cp.tile(txL[:, kgamma][None,:], (ny,1)) # [ny,nx]
            ttxH = cp.tile(txH[:, kgamma][None,:], (ny,1)) # [ny,nx]
            ttyL = cp.tile(tyL[:, kgamma][:,None], (1,nx)) # [ny,nx]
            ttyH = cp.tile(tyH[:, kgamma][:,None], (1,nx)) # [ny,nx]
            # dsclip = overlap corresponding to the lower beam edge - overlap of corresponding to the upper beam edge
            sclipH = aux_calculate_overlap_between_ray_pixel_FB_Curve_clip_GPU(ttxL, ttxH, ttyL, ttyH, costheta[kgamma], sintheta[kgamma], xxedge, yyedge, ax, ay, dxy)
            dsclip = sclipL - sclipH
            
            # axx.clear()
            # axx.set_title(f'{kla} - {kgamma}')
            # plt.imshow(dsclip, cmap='gray')
            # plt.axis('image')
            # plt.axis('off')
            # plt.pause(0.01)

            # Contribution only from intersected regions
            iik = dsclip > 0
            spd = cp.sqrt((ax - xx[iik])**2 + (ay - yy[iik])**2) # weighted by source to pixel center distance
            g[kla,kgamma - 1] = cp.sum(im[iik] * dsclip[iik] / spd) / dgamma[kgamma - 1] # divide by the beamlet fan angle width, channel variant

            sclipL = sclipH

        if kla % 10 == 0:
            print(f'Completed view #{kla + 1}')
    
    return g        

'''------------------------------------------------------------------------
[F] - perform_FBFBP_Flat_FS_Ramp
[I] - slice: img.Slice2D object containing image slice information and data
[I] - alpha: apodization coefficient for ramp filter [0.5, 1)
[I] - g: fanogram of size [nlambda, nu]
[I] - R: source to isocenter distance
[I] - D: source to detector distance
[I] - lambdai,dlambda: view angle grid definition, lambdak = lambdas + k * dlambda
[I] - uoff,du: detector channel grid definition, uk = (k - uoff) * du
[I] - fn_recon_data: file name to save the reconstructed image
[N] - Perform conventional FBP reconstruction for fan-beam geometry 
        Flat geometry
        Full scan (0.5 weighting)
        Ramp filter
[H] - 20250425 created by ZYu
------------------------------------------------------------------------'''
def perform_FBFBP_Flat_FS_Ramp(slice:img.Slice2D,alpha,g,R,D,lambdai,dlambda,uoff,du,fn_recon_data = None):                
                 
        # Step 1: Apply FB weighting: gFB = D/sqrt(uu+DD)*g0
        nlambda,nu = g.shape
        u =  (np.arange(nu) - uoff) * du
        uu = np.ones([nlambda,1]) * u
        gFB = D*g/np.sqrt(uu**2 + D**2) # [nlambda,nu]
        
        # Step 2: Apply ramp filter
        gF = apply_ramp_filter_rowwise_flat(gFB, alpha, du)
        
        # Step 3: Perform backprojection        
        perform_BP_FB_Flat(gF,slice,uoff,du,lambdai,dlambda,R,D)
        
        # Save the reconstructed image
        if fn_recon_data is not None:
            slice.write_header(fn_recon_data)
            slice.write_slice(fn_recon_data)

'''============================================================================
Class       
    recon_FB_Flat_FS
Parent
    n.a.
Attributes
    fn_recon_config: str, file name of the reconstruction configuration XML file
    device_name: str, name of the device
    xml_version: str, version of the XML configuration
    raw_data_path: str, path to the raw data
    recon_data_folder: str, folder for saving reconstruction data
    slice_colume_number: int, number of columns in the slice
    slice_colume_space_mm: float, space between columns in mm
    slice_colume_offset: float, offset for column position in mm
    slice_row_number: int, number of rows in the slice
    slice_row_space_mm: float, space between rows in mm
    slice_row_offset: float, offset for row position in mm
    slice_thickness_mm: float, thickness of each slice in mm
    slope: float, slope for pixel value scaling
    intercept: float, intercept for pixel value scaling
    reconstruction_method: str, method used for reconstruction (e.g., 'FBFBP_Flat_FS_Ramp')
    apodization_coef: float, coefficient for apodization in ramp filter
Methods
    __init__(fn_recon_config): Initializes the class and loads the reconstruction configuration.
    load_recon_config(fn_recon_config): Loads the reconstruction configuration from an XML file.
    perform_recon_FBP_Ramp(slice, alpha, raw_data_path, recon_data_folder): Performs FBP reconstruction with ramp filter. 
Notes
    
History: 202505 created by ZYu
============================================================================'''
class recon_FB_Flat():
    def __init__(self, fn_recon_config):
        self.fn_recon_config = fn_recon_config
        self.load_recon_config(fn_recon_config)    
        match self.reconstruction_method:
            case 'FBFBP_Flat_FS_Ramp':
                self.perform_recon_FBP_Ramp()
            case _:
                raise ValueError(f"Unknown reconstruction method: {self.reconstruction_method}")   
             
    '''============================================================================
    Load the reconstruction configuration from an XML file.
    [H] - 202505 created by ZYu
    ============================================================================'''
    def load_recon_config(self,fn_recon_config=None):
        # Load the reconstruction configuration from the XML file
        if fn_recon_config is None:
            fn_recon_config = self.fn_recon_config

        root = ET.parse(fn_recon_config)
        XSS = root.getroot()
        
        # General info
        #------ general
        self.device_name = XSS.find('device_name').text
        self.xml_version = XSS.find('xml_version').text
        self.raw_data_path = XSS.find('raw_data_path').text
        self.recon_data_folder = XSS.find('recon_data_folder').text
        
        # reconstruction geometry
        slice_definition = XSS.find('slice_definition')
        self.slice_colume_number = np.int32(slice_definition.find('slice_colume_number').text)
        self.slice_colume_space_mm = np.float32(slice_definition.find('slice_colume_space_mm').text)
        self.slice_colume_offset = np.float32(slice_definition.find('slice_colume_offset').text) 
        self.slice_row_number = np.int32(slice_definition.find('slice_row_number').text)
        self.slice_row_space_mm = np.float32(slice_definition.find('slice_row_space_mm').text)
        self.slice_row_offset = np.float32(slice_definition.find('slice_row_offset').text)
        self.slice_thickness_mm = np.float32(slice_definition.find('slice_thickness_mm').text)
        self.slope = np.float32(slice_definition.find('slope').text)
        self.intercept = np.float32(slice_definition.find('intercept').text)
        
        # Recon configurations
        recon_config = XSS.find('recon_configurations')
        self.reconstruction_method = recon_config.find('reconstruction_method').text
        self.apodization_coef = np.float32(recon_config.find('apodization_coef').text)
    
    '''============================================================================
    [F] - Perform FBP reconstruction with ramp filter.
    [I] - slice: img.Slice2D object containing image slice information and data
    [I] - alpha: apodization coefficient for ramp filter [0.5, 1)
    [I] - raw_data_path: str, path to the raw data file
    [I] - recon_data_folder: str, folder for saving reconstruction data
    [O] - slice: img.Slice2D object containing the reconstructed image
    [H] - 202505 created by ZYu
    ============================================================================'''    
    def perform_recon_FBP_Ramp(self, slice = None, alpha = None, raw_data_path = None, recon_data_folder = None):
        #----------------------------------------------------------------------------------------
        # Address inputs
        if slice is None:
            slice = img.Slice2D(self.slice_colume_number, self.slice_row_number, self.slice_colume_space_mm, self.slice_row_space_mm, self.slice_colume_offset, self.slice_row_offset,
                            self.slice_thickness_mm, self.slope, self.intercept)
        if alpha is None:
            alpha = self.apodization_coef
        
        if raw_data_path is None:
            raw_data_path = self.raw_data_path
            
        if recon_data_folder is None:
            recon_data_folder = self.recon_data_folder
        
        # Create recon data path
        fn_recon_data = f"FBFPB_Flat_FS_Ramp_alpha{self.apodization_coef:.2f}_{self.slice_row_number}x{self.slice_colume_number}.img"
        fn_recon_data = os.path.join(recon_data_folder, fn_recon_data)
                
        #----------------------------------------------------------------------------------------
        # Perform reconstruction
        fanogram_header  = fio.read_header(raw_data_path)
        match fanogram_header['header_type_string']:
            case enm.HeaderType.PROJECTION_BASIC.name:
                lambdai = fanogram_header['starting_angular_position']
                dlambda = fanogram_header['angular_space_per_view']
                uoff = fanogram_header['ochannel']
                du = fanogram_header['dchannel']
                R = fanogram_header['SID']
                D = fanogram_header['SDD']
            case _:
                raise ValueError(f"Unknown header type: {fanogram_header['header_type_string']}")
        
        g0 = fio.read_volume(raw_data_path) #[nlambda,nu]
        g0 = g0.squeeze()
        
        perform_FBFBP_Flat_FS_Ramp(slice,alpha,g0,R,D,lambdai,dlambda,uoff,du,fn_recon_data)
        
        fig = plt.figure()
        ax = fig.add_subplot()
        im = slice.slice
        ax.clear()
        ax.set_title(f'Reconstructed Image: FBP Ramp')
        ax.imshow(im, cmap='gray')
        plt.axis('image')
        plt.axis('off')
        plt.pause(0.01)
               
        return slice
        
        
        