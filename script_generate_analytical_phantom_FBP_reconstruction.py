import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from XSSModules import filtered_backprojection as bp
import os
from numpy.fft import fft, ifft, fft2, ifft2, fftshift, fftfreq
from skimage.data import shepp_logan_phantom
from skimage.transform import radon

matplotlib.use('TkAgg')

if __name__ == "__main__":

    # curved fan-beam test
    phantom = np.fromfile('Data/Tire3D_H44-5X16-5R21_LAC_Partial_1024x1x1024.img', dtype=np.float32, offset=48)
    phantom = phantom.reshape((1024, 1024))
    phantom = phantom[:, ::-1]
    sino_path = './Data/Sinograms/Tire_phantom_Helical_CurveD_sinogram_3600view_4065channel.npy'
    sino = np.load(sino_path)
    gamma_num, lam_num = sino.shape
    img_size = 1024
    fov = 1280 # 1280 / 3
    dgamma = np.float64(0.03)
    dgamma_rad = np.deg2rad(dgamma)
    dlam = np.float64(360 / lam_num)
    dlam_rad = np.deg2rad(dlam)
    SID = 415.75 # 415.75
    pad_length = np.power(2, 14)
    delta = 1.0 / dgamma_rad
    # filter = bp.generate_filter_frequency_domain(gamma_num, "hamming", delta, gamma_num)
    filter = bp.generate_filter_spatial_domain(gamma_num, 0.5, delta)
    x = np.linspace(0, 2*gamma_num-1, 2*gamma_num-1)
    plt.figure()
    plt.plot(x, filter)
    plt.show()
    recon = bp.Curved_Fan_Beam_Backprojection(sino, fov, img_size, dlam_rad, dgamma_rad, filter, pad_length, SID)
    diff = phantom - recon
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    im0 = axes[0].imshow(sino, cmap="gray")
    im1 = axes[1].imshow(phantom, vmin=0.02, vmax=0.03, cmap="gray")
    im2 = axes[2].imshow(recon, vmin=0.02, vmax=0.03, cmap="gray")
    im3 = axes[3].imshow(diff, vmin=-0.001, vmax=0.001, cmap="rainbow")
    fig.colorbar(im1, ax=axes[1])
    fig.colorbar(im2, ax=axes[2])
    fig.colorbar(im3, ax=axes[3])
    plt.show()
    a = np.linspace(0, img_size, img_size, endpoint=False)
    plt.figure()
    #plt.plot(a, phantom[740, :], 'g', a, recon[740, :], 'r')
    plt.plot(a, recon[740, :] - phantom[740, :], 'r')
    plt.show()
    '''
    np.save("C:/Users/jzhan685/CT/Codes/XSS/Recon Results/Tire3D_H44-5X16_5R21/phantom_Rolical_3600view_3065channel.npy", phantom)
    np.save("C:/Users/jzhan685/CT/Codes/XSS/Recon Results/Tire3D_H44-5X16_5R21/recon_Rolical_3600view_3065channel.npy", recon)
    np.save("C:/Users/jzhan685/CT/Codes/XSS/Recon Results/Tire3D_H44-5X16_5R21/diff_Rolical_3600view_3065channel.npy", diff)
    '''







    '''
    # flat panel fan-beam test
    phantom = np.fromfile('./Data/FORBILD_head_static_LAC_1024x1024x1.img', dtype=np.float32, offset=48)
    phantom = phantom.reshape((1024, 1024))
    # phantom = phantom[::-1, :]
    sino_path = './JZhang_0927/FORBILD_phantom_sinogram_720view_633channel.npy'
    sino = np.load(sino_path)
    channel_num, lam_num = sino.shape
    img_size = 1024
    fov = 283
    dchannel = np.float64(0.9)
    dlam = np.float64(360 / lam_num)
    dlam_rad = np.deg2rad(dlam)
    SID = 800
    SDD = 1200
    pad_length = np.power(2, 14)
    delta = 1.0 / dchannel
    filter = bp.generate_filter_frequency_domain(channel_num, "hamming", delta, pad_length)
    x = np.linspace(0, pad_length, pad_length)
    plt.figure()
    plt.plot(x, filter)
    plt.show()
    recon = bp.Flat_Panel_Fan_Beam_Backprojection(sino, fov, img_size, dlam_rad, dchannel, filter, pad_length, SID, SDD)
    diff = phantom - recon
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    im0 = axes[0].imshow(sino, cmap="gray")
    im1 = axes[1].imshow(phantom, vmin=1, vmax=1.2, cmap="gray")
    im2 = axes[2].imshow(recon, vmin=1, vmax=1.2, cmap="gray")
    im3 = axes[3].imshow(diff, vmin=-0.01, vmax=0.01, cmap="rainbow")
    fig.colorbar(im1, ax=axes[1])
    fig.colorbar(im2, ax=axes[2])
    fig.colorbar(im3, ax=axes[3])
    plt.show()
    a = np.linspace(0, img_size, img_size, endpoint=False)
    plt.figure()
    plt.plot(a, recon[740, :] - phantom[740, :], 'r')
    plt.show()
    '''
    '''
    # parallel-beam test
    phantom_path = './JZhang_0927/phantom_shepp_logan.npy'
    sino_path = 'JZhang_0927/sinogram_shepp_logan_720view.npy'
    phantom = np.load(phantom_path)
    sino = np.load(sino_path)
    mask = phantom > 0
    mask[100:300, 100:300] = 1
    channel_num, angle_num = sino.shape
    img_size = phantom.shape[0]
    fov = img_size
    dtheta = 180 / angle_num
    delta = img_size / fov
    pad_length = np.power(2, 13)
    filter = bp.generate_filter_frequency_domain(channel_num, "hamming", delta, pad_length)
    x = np.linspace(0, pad_length, pad_length, endpoint=False)
    plt.figure()
    plt.plot(x, filter)
    plt.show()
    recon = bp.Parallel_Beam_Backprojection(sino, fov, img_size, dtheta, delta, filter, pad_length)
    diff = (phantom - recon)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    im0 = axes[0].imshow(sino, cmap="gray")
    im1 = axes[1].imshow(phantom, vmin=0, vmax=1, cmap="gray")
    im2 = axes[2].imshow(recon, vmin=0, vmax=1, cmap="gray")
    im3 = axes[3].imshow(diff, vmin=-0.001, vmax=0.001, cmap="rainbow")
    fig.colorbar(im1, ax=axes[1])
    fig.colorbar(im2, ax=axes[2])
    fig.colorbar(im3, ax=axes[3])
    plt.show()
    a = np.linspace(0, img_size, img_size, endpoint=False)
    plt.figure()
    plt.plot(a, phantom[226, :], 'g', a, recon[226, :], 'r')
    plt.show()
    '''
