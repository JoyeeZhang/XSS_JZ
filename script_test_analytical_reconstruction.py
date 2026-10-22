import os
import numpy as np
from XSSModules.analytical_recon import FBFPD
from XSSModules import file_io as fio
from XSSModules import recon_support as rs

project_folder = os.path.dirname(os.path.abspath(__file__))
phantom_path = os.path.join(project_folder, 'Data', 'FORBILD_head_static_LAC_Full_256x256x1.img')
projection_path = os.path.join(project_folder, 'Data', 'FORBILD_head_static_LAC_465x1x360.prj')

phantom_header = fio.read_header(phantom_path)
phantom = fio.read_volume(phantom_path).squeeze().astype(np.float64)
projection_header = fio.read_header(projection_path)
projection = fio.read_volume(projection_path).astype(np.float64)

incident_photon_count = 1e7
water_lac = 0.02  # Assumed water LAC in mm^-1 at the effective scan energy
photon_count_label = f'{incident_photon_count:.0e}'

# Beer-Lambert photon statistics require physical line integrals. Convert the
# relative-LAC projection to physical attenuation for noise generation, then
# return it to the relative-LAC domain used by the phantom and reconstruction.
physical_projection = water_lac * projection
noisy_projection = rs.add_poisson_noise(
    physical_projection,
    N0=incident_photon_count,
)
noisy_projection /= water_lac

reconstructor = FBFPD(
    projection_header=projection_header,
    projection=noisy_projection,
    image_shape=phantom.shape,
    pixel_size=(phantom_header['drow'], phantom_header['dcol']),
    filter_type='hann',
)
reconstruction = reconstructor.filtered_backprojection()
reconstruction_path = os.path.join(
    project_folder,
    'Data',
    f'FBFPD_noisy_N0_{photon_count_label}_reconstruction_'
    f'{reconstruction.shape[1]}x{reconstruction.shape[0]}x1.img',
)
rs.save_reconstruction_image(reconstruction, reconstruction_path, phantom_header)

print(f'FBFPD reconstruction shape: {reconstruction.shape}')
print(f'Saved reconstruction: {reconstruction_path}')
rs.show_reconstruction_image(
    phantom,
    reconstruction,
    reconstruction_title=f'Noisy FBFPD reconstruction ($N_0={photon_count_label}$)',
)
