import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

'''
phantom = np.fromfile('./Data/Tire3D_H44-5X16-5R21_LAC_Partial_1024x1x1024.img', dtype=np.float32, offset=48)
phantom = phantom.reshape((1024, 1024))
phantom = phantom[:, ::-1]
mu_water = 0.0184
L = 180
W = 400
phantom_HU = 1000 * (phantom - mu_water) / mu_water
plt.figure()
plt.imshow(phantom_HU, vmin=-20, vmax=380, cmap="gray")
plt.colorbar()
plt.show()

plt.imsave(
    "C:/Users/jzhan685/CT/2026 CT meeting/Figures/phantom_tube.png",
    phantom_HU,
    cmap="gray",
    vmin=L-W/2,
    vmax=L+W/2
)



sino_path = './JZhang_0927/TireH445_phantom_sinogram_1920view_865gamma.npy'
sino = np.load(sino_path)
plt.figure()
plt.imshow(sino, vmin=-20, vmax=380, cmap="gray")
plt.colorbar()
plt.show()

plt.imsave(
    "C:/Users/jzhan685/CT/2026 CT meeting/Figures/sino_tube_1920.png",
    sino,
    cmap="gray"
)



recon_path = 'C:/Users/jzhan685/CT/2026 CT meeting/Data/recon_tube_1920.npy'
recon = np.load(recon_path)
recon_HU = 1000 * (recon - mu_water) / mu_water
plt.figure()
plt.imshow(recon_HU, vmin=-20, vmax=380, cmap="gray")
plt.colorbar()
plt.show()

plt.imsave(
    "C:/Users/jzhan685/CT/2026 CT meeting/Figures/recon_tube_1920.png",
    recon_HU,
    cmap="gray",
    vmin=L-W/2,
    vmax=L+W/2
)



diff_path = 'C:/Users/jzhan685/CT/2026 CT meeting/Data/diff_tube_1920.npy'
diff = np.load(diff_path)
diff_HU = 1000 * (diff - mu_water) / mu_water

plt.imsave(
    "C:/Users/jzhan685/CT/2026 CT meeting/Figures/diff_tube_1920.png",
    diff_HU,
    cmap="seismic",
    vmin=L-W/2,
    vmax=L+W/2
)

'''

'''
sino_path1 = './JZhang_0927/TireH445_Simplify4_phantom_sinogram_360view_865gamma.npy'
sino1 = np.load(sino_path1)
sino_path2 = './JZhang_0927/TireH445_Simplify4_phantom_sinogram_360view_865gamma_nproots.npy'
sino2 = np.load(sino_path2)
diff1 = sino2 - sino1
plt.figure(figsize=[10, 3])
plt.subplot(131)
plt.imshow(sino1, cmap="gray")
plt.subplot(132)
plt.imshow(sino2, cmap="gray")
plt.subplot(133)
plt.imshow(diff1, vmin=-1e-12, vmax=1e-12, cmap="rainbow")
plt.colorbar()
plt.show()
'''

# results comparison
recon_path = './Recon Results/Tire3D_H44-5X16_5R21/recon_Helical_3600view_4065channel.npy'
phantom_path = './Recon Results/Tire3D_H44-5X16_5R21/phantom_Helical_3600view_4065channel.npy'
# recon_path = './JZhang_0927/Tire_results/recon_SingleTube4_3600view.npy'
# phantom_path = './JZhang_0927/Tire_results/phantom_SingleTube4_3600view.npy'
recon = np.load(recon_path)
phantom = np.load(phantom_path)
mu_water = 0.0184
L = 180
W = 400
phantom_HU = 1000 * (phantom - mu_water) / mu_water
recon_HU = 1000 * (recon - mu_water) / mu_water
diff_HU = recon_HU - phantom_HU
diff = recon - phantom
plt.figure(figsize=[20, 5])
plt.subplot(141)
plt.imshow(phantom_HU, vmin=L-W/2, vmax=L+W/2, cmap="gray")
plt.subplot(142)
plt.imshow(recon_HU, vmin=L-W/2, vmax=L+W/2, cmap="gray")
plt.subplot(143)
plt.imshow(diff_HU, vmin=-10, vmax=10, cmap="seismic")
plt.subplot(144)
plt.imshow(diff, vmin=-0.0001, vmax=0.0001, cmap="seismic")
plt.colorbar()
'''
plt.imsave(
    "C:/Users/jzhan685/CT/2026 CT meeting/Figures/recon_tube_3600.png",
    recon_HU,
    cmap="gray",
    vmin=L-W/2,
    vmax=L+W/2
)

plt.imsave(
    "C:/Users/jzhan685/CT/2026 CT meeting/Figures/phantom_tube_3600.png",
    phantom_HU,
    cmap="gray",
    vmin=L-W/2,
    vmax=L+W/2
)

plt.imsave(
    "C:/Users/jzhan685/CT/2026 CT meeting/Figures/diff_tube_3600.png",
    diff_HU,
    cmap="seismic",
    vmin=-10,
    vmax=10
)
'''

img_size = phantom.shape[0]
axis = np.linspace(0, img_size, img_size, endpoint=False)
'''
column_num0 = 810
row_num1 = 512
column_num1 = 856
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(axis, phantom_HU[row_num1, :], 'blue', axis, recon_HU[row_num1, :], 'red', linewidth=0.5)
ax[0].set_ylim(L-W/2, L+W/2)
ax[0].set_xlim(0, 1024)
ax[0].set_ylabel("HU")
ax[1].plot(axis, phantom_HU[:, column_num0], 'blue', axis, recon_HU[:, column_num0], 'red', linewidth=0.5)
ax[1].set_ylim(L-W/2, L+W/2)
ax[1].set_xlim(0, 1024)
ax[1].set_ylabel("HU")
ax[2].plot(axis, phantom_HU[:, column_num1], 'blue', axis, recon_HU[:, column_num1], 'red', linewidth=0.5)
ax[2].set_ylim(L-W/2, L+W/2)
ax[2].set_xlim(0, 1024)
ax[2].set_ylabel("HU")
plt.tight_layout()
plt.show()
'''
'''
column_num1 = 810
plt.figure(figsize=(10, 4))
plt.plot(axis, phantom_HU[:, column_num1], 'blue', axis, recon_HU[:, column_num1], 'red', linewidth=0.5)
plt.ylim(L-W/2, L+W/2)
plt.xlim(0, 1024)
plt.ylabel("HU")
plt.tight_layout()
plt.show()
'''
row_num1 = 512
plt.figure(figsize=(10, 4))
plt.plot(axis, phantom_HU[row_num1, :], 'blue', axis, recon_HU[row_num1, :], 'red', linewidth=0.5)
plt.ylim(L-W/2, L+W/2)
plt.xlim(0, 1024)
plt.ylabel("HU")
plt.tight_layout()
plt.show()

'''
sino_path = './Data/Sinograms/Tire_phantom_Rolical_CurveD_sinogram_3600view_3065channel.npy'
sino = np.load(sino_path)
plt.imsave(
    "C:/Users/jzhan685/CT/2026 CT meeting/Figures/sino_tube_3600.png",
    sino,
    cmap="gray"
)
'''

'''
fig, ax = plt.subplots(figsize=(0.5, 6))
L = 180
W = 400

norm = matplotlib.colors.Normalize(vmin=L-W/2, vmax=L+W/2)

cbar = matplotlib.colorbar.ColorbarBase(
    ax,
    cmap='gray',
    norm=norm,
    orientation='vertical'
)

cbar.ax.tick_params(color='white', labelcolor='white')

# Border color
cbar.outline.set_edgecolor('white')

plt.savefig('C:/Users/jzhan685/CT/2026 CT meeting/Figures/gray_colorbar_only.png', dpi=300, bbox_inches='tight', transparent=True)
plt.close()
'''




