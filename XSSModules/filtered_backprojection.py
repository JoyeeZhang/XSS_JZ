import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft, ifft, fftshift, ifftshift

def generate_filter_frequency_domain(channel_num, filter_type="ram-lak", delta=1.0, pad_length=1024):
    val = np.arange(pad_length, dtype=np.float64) / pad_length - 0.5
    f = np.abs(val * delta)

    if filter_type == "ram-lak":
        filter = f
    elif filter_type == "shepp-logan":
        filter = f * (np.sinc(2 * val))
    elif filter_type == "hamming":
        filter = f * (0.54 + 0.46 * np.cos(2 * np.pi * val))
    elif filter_type == "hann":
        filter = f * (0.5 + 0.5 * np.cos(2 * np.pi * val))
    else:
        raise ValueError("Unknown filter type")

    return filter

def gamma(t):
    t = np.asarray(t)
    idx0 = np.abs(t) < 1e-12
    idx1 = ~idx0
    y = np.empty_like(t)
    y[idx0] = 0.5
    y[idx1] = np.sin(t[idx1]) / t[idx1] + (np.cos(t[idx1]) - 1) / (t[idx1] ** 2)

    return y

def generate_filter_spatial_domain(channel_num, alpha, delta=1.0):
    val = np.pi * np.arange(-channel_num + 1, channel_num)
    c
    filter = 0.5 * filter * (delta**2)

    return filter


def Parallel_Beam_Backprojection(sinogram, fov, img_size, dtheta, delta, filter, pad_length):
    channel_num, angle_num = sinogram.shape
    start = (pad_length - channel_num) // 2

    # filter sinogram
    sino_pad = np.zeros((pad_length, angle_num), dtype=np.float64)
    sino_pad[start:start+channel_num, :] = sinogram
    sino_fft = fft(sino_pad, axis=0)
    sino_filtered = np.real(ifft(sino_fft * fftshift(filter[:, np.newaxis]), axis=0))
    sino_filtered = sino_filtered[start:start+channel_num, :]

    # define parameters
    channel_offset = (channel_num - 1) / 2.0
    theta_array = np.deg2rad((np.arange(angle_num) * dtheta))
    dtheta_rad = np.deg2rad(dtheta)
    img_offset = (img_size - 1) / 2.0
    pixel_size = fov / img_size
    index = np.arange(img_size) - img_offset
    X0 = pixel_size * index[np.newaxis, :]
    Y0 = pixel_size * index[:, np.newaxis]

    recon = np.zeros((img_size, img_size), dtype=np.float64)
    # filenames = []

    for itheta, theta in enumerate(theta_array):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        T_theta = (X0 * cos_theta - Y0 * sin_theta) * delta + channel_offset
        T_floor = np.floor(T_theta).astype(np.int64)
        alpha = T_theta - T_floor

        mask0 = (T_floor >= 0) & (T_floor < channel_num)
        mask1 = (T_floor + 1 >= 0 ) & (T_floor + 1 < channel_num)

        map = np.zeros_like(T_theta, dtype=np.float64)

        if np.any(mask0):
            index0 = np.clip(T_floor, 0, channel_num - 1)
            map[mask0] += (1.0 - alpha[mask0]) * sino_filtered[index0[mask0], itheta]
        if np.any(mask1):
            index1 = np.clip(T_floor + 1, 0, channel_num - 1)
            map[mask1] += alpha[mask1] * sino_filtered[index1[mask1], itheta]

        recon += map
        '''
        plt.imshow(recon, cmap='gray', aspect='auto')
        plt.axis('off')
        filename = f'frame_{itheta:03d}.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        filenames.append(filename)
        '''

    recon *= dtheta_rad

    return recon


def Curved_Fan_Beam_Backprojection(sinogram, fov, img_size, dlam_rad, dgamma_rad, filter, pad_length, SID):
    gamma_num, lam_num = sinogram.shape
    gamma_offset = (gamma_num - 1) / 2.0
    gamma_array1 = (gamma_offset - np.arange(gamma_num)) * dgamma_rad
    gamma_array2 = dgamma_rad * np.arange(-gamma_num + 1, gamma_num)
    weights = np.ones_like(gamma_array2)
    mask = (np.abs(gamma_array2) > 1e-12)
    weights[mask] = (gamma_array2[mask] / np.sin(gamma_array2[mask])) ** 2

    # filter sinogram
    weighted_sino = sinogram * np.cos(gamma_array1)[:, np.newaxis]
    start1 = (pad_length - gamma_num) // 2
    sino_pad = np.zeros((pad_length, lam_num), dtype=np.float64)
    sino_pad[start1:start1 + gamma_num, :] = weighted_sino
    sino_fft = np.fft.fft(sino_pad, axis=0)
    weighted_filter = weights * filter
    filter_pad = np.zeros((pad_length, 1), dtype=np.float64)
    filter_pad[:gamma_num, 0] = weighted_filter[gamma_num - 1:2 * gamma_num]
    filter_pad[-gamma_num + 1:, 0] = weighted_filter[:gamma_num - 1]
    sino_filtered = np.real(ifft(sino_fft * fft(filter_pad, axis=0) * dgamma_rad, axis=0))
    sino_filtered = sino_filtered[start1:start1 + gamma_num, :]

    '''
    gamma_num, lam_num = sinogram.shape
    gamma_offset = (gamma_num - 1) / 2.0
    gamma_array = (gamma_offset - np.arange(gamma_num)) * dgamma_rad
    weights = np.ones_like(gamma_array)
    mask = (np.abs(gamma_array) > 1e-12)
    weights[mask] = (gamma_array[mask] / np.sin(gamma_array[mask])) ** 2

    # filter sinogram

    weighted_sino = sinogram * np.cos(gamma_array[:, np.newaxis])
    start = (pad_length - gamma_num) // 2
    sino_pad = np.zeros((pad_length, lam_num), dtype=np.float64)
    sino_pad[start:start+gamma_num, :] = weighted_sino
    sino_fft = np.fft.fft(sino_pad, axis=0)
    ifft_filter = np.real(np.fft.ifft(filter, axis=0))
    weighted_filter = ifft_filter * weights
    filter_pad = np.zeros((pad_length, 1), dtype=np.float64)
    filter_pad[:gamma_num//2, 0] = weighted_filter[:gamma_num//2]
    filter_pad[-gamma_num//2:, 0] = weighted_filter[-gamma_num//2:]
    sino_filtered = np.real(ifft(sino_fft * fft(filter_pad, axis=0) * dgamma_rad, axis=0))
    sino_filtered = sino_filtered[start:start+gamma_num, :]
    '''

    # define parameters
    lam_array = np.arange(lam_num) * dlam_rad
    img_offset = (img_size - 1) / 2.0
    pixel_size = np.float64(fov / img_size)
    index = np.arange(img_size) - img_offset
    X0 = pixel_size * index[np.newaxis, :]
    Y0 = pixel_size * index[::-1, np.newaxis]

    # back projection
    # filenames = []
    recon = np.zeros((img_size, img_size), dtype=np.float64)
    for ilam, lam in enumerate(lam_array):
        detector = sino_filtered[:, ilam]
        sin_lam = np.sin(lam)
        cos_lam = np.cos(lam)

        X_lam = X0 * cos_lam - Y0 * sin_lam
        Y_lam = X0 * sin_lam + Y0 * cos_lam
        T_lam = gamma_offset - np.arctan2(Y_lam, (SID - X_lam)) / dgamma_rad

        T_floor = np.floor(T_lam).astype(np.int64)
        T_ceil = T_floor + 1
        alpha = T_lam - T_floor

        T_floor_clip = np.clip(T_floor, 0, gamma_num - 1)
        T_ceil_clip = np.clip(T_ceil, 0, gamma_num - 1)

        map = (1 - alpha) * detector[T_floor_clip] + alpha * detector[T_ceil_clip]

        recon += (SID / ((SID - X_lam) ** 2 + Y_lam ** 2)) * map
        '''
        plt.imshow(recon, cmap='gray', aspect='auto')
        plt.axis('off')
        filename = f'./Recon_steps_2_1/frame_{ilam:03d}.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        filenames.append(filename)
        '''

    recon *= dlam_rad / 2.0

    return recon

def Flat_Panel_Fan_Beam_Backprojection(sinogram, fov, img_size, dlam_rad, dchannel, filter, pad_length, SID, SDD):
    channel_num, lam_num = sinogram.shape
    channel_offset = (channel_num - 1) / 2.0
    channel_array = (channel_offset - np.arange(channel_num)) * dchannel
    weights = SDD ** 2 / np.sqrt(SDD ** 2 + channel_array ** 2)
    weighted_sino = sinogram * weights[:, np.newaxis]

    # filter sinogram
    start = (pad_length - channel_num) // 2
    sino_pad = np.zeros((pad_length, lam_num), dtype=np.float64)
    sino_pad[start:start+channel_num, :] = weighted_sino
    sino_fft = np.fft.fft(sino_pad, axis=0)
    sino_filtered = np.real(ifft(sino_fft * fftshift(filter[:, np.newaxis]), axis=0))
    sino_filtered = sino_filtered[start:start+channel_num, :]

    # define parameters
    lam_array = np.arange(lam_num) * dlam_rad
    img_offset = (img_size - 1) / 2.0
    pixel_size = np.float64(fov / img_size)
    index = np.arange(img_size) - img_offset
    X0 = pixel_size * index[np.newaxis, :]
    Y0 = pixel_size * index[::-1, np.newaxis]

    filenames = []
    recon = np.zeros((img_size, img_size), dtype=np.float64)
    for ilam, lam in enumerate(lam_array):
        detector = sino_filtered[:, ilam]
        sin_lam = np.sin(lam)
        cos_lam = np.cos(lam)

        X_lam = X0 * cos_lam - Y0 * sin_lam
        Y_lam = X0 * sin_lam + Y0 * cos_lam
        T_lam = channel_offset - (Y_lam / (SID - X_lam) * SDD)

        T_floor = np.floor(T_lam).astype(np.int64)
        T_ceil = T_floor + 1
        alpha = T_lam - T_floor

        T_floor_clip = np.clip(T_floor, 0, channel_num - 1)
        T_ceil_clip = np.clip(T_ceil, 0, channel_num - 1)

        map = (1 - alpha) * detector[T_floor_clip] + alpha * detector[T_ceil_clip]

        recon += (SID / (SID - X_lam) ** 2) * map
        '''
        plt.imshow(recon, cmap='gray', aspect='auto')
        plt.axis('off')
        filename = f'./Recon_steps_2_1/frame_{ilam:03d}.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        filenames.append(filename)
        '''

    recon *= dlam_rad / 2.0

    return recon





