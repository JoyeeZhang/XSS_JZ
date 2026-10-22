import os
from functools import partial
import numpy as np
from XSSModules.analytical_recon import FBFPD
from XSSModules import file_io as fio
from XSSModules import recon_support as rs
from XSSModules.iterative_recon import (
    PWLSReconstructionL1FISTA,
    PWLSReconstructionL2CG,
    PWLSReconstructionL2GD,
    PWLSReconstructionL2LBFGS,
)
from XSSModules.numerical_projector import FBFPJosephProjector

project_folder = os.path.dirname(os.path.abspath(__file__))
phantom_path = os.path.join(project_folder, 'Data', 'FORBILD_head_static_LAC_Full_256x256x1.img')
projection_path = os.path.join(project_folder, 'Data', 'FORBILD_head_static_LAC_465x1x360.prj')

number_of_iterations = 100
beta = 5e-1
objective_tolerance = 1e-3
incident_photon_count = 1e5
add_noise_option = False
initial_image_option = 'zeros'  # Options: 'zeros' or 'fbp'
optimizer_label = 'FISTA'  # Options: 'GD', 'CG', 'LBFGS', or 'FISTA'
photon_count_label = f'{incident_photon_count:.0e}'
data_label = (
    f'noisy_N0_{photon_count_label}'
    if add_noise_option
    else ''
)
optimization_label = (
    'L1_FISTA'
    if optimizer_label == 'FISTA'
    else f'L2_{optimizer_label}'
)

fbp_reconstruction_path = os.path.join(project_folder, 'Data', f'FBFPD_{data_label}_256x256x1.img')
iteration_save_folder = os.path.join(project_folder,
                                     'Data',
                                     f'PWLS_{data_label}_{optimization_label}_'
                                     f'{initial_image_option}_iterations')
metrics_path = os.path.join(iteration_save_folder, 'PWLS_metrics.csv')

phantom_header = fio.read_header(phantom_path)
phantom = fio.read_volume(phantom_path).squeeze().astype(np.float64)
reconstruction_path = os.path.join(project_folder, 'Data',
                                   f'PWLS_{data_label}_{optimization_label}_'
                                   f'{initial_image_option}_iterations_'
                                   f'{phantom.shape[1]}x'
                                   f'{phantom.shape[0]}x1.img')

if os.path.isfile(reconstruction_path) and os.path.isfile(metrics_path):
    print('Saved iterative reconstruction and iteration process found.')
    print(f'Loading reconstruction: {reconstruction_path}')
    reconstruction = fio.read_volume(reconstruction_path).squeeze().astype(np.float64)
else:
    projection_header = fio.read_header(projection_path)
    projection = fio.read_volume(projection_path).astype(np.float64)
    nview, nrow, nchannel = projection.shape
    middle_row = nrow // 2
    line_integrals = projection[:, middle_row, :]
    if add_noise_option:
        line_integrals = rs.add_poisson_noise(
            line_integrals,
            N0=incident_photon_count,
        )
        print(f'Added Poisson noise with N0={photon_count_label}.')
    else:
        print('Using noise-free line integrals.')

    projector = FBFPJosephProjector(
        image_shape=phantom.shape,
        pixel_size=(phantom_header['drow'], phantom_header['dcol']),
        lambda_start=float(projection_header['starting_angular_position']),
        dlambda=float(projection_header['angular_space_per_view']),
        nview=nview,
        channel_offset=float(projection_header['ochannel']),
        channel_spacing=float(projection_header['dchannel']),
        nchannel=nchannel,
        SID=float(projection_header['SID']),
        SPD=float(projection_header['SDD']),
        beam_upsampling=1)

    _, _, adjoint_error = projector.adjoint_test(seed=0)
    if adjoint_error > 1e-10:
        raise RuntimeError(f"projector adjoint test failed: {adjoint_error}")

    process_iteration = partial(rs.process_iteration,
                                ground_truth=phantom,
                                save_folder=iteration_save_folder,
                                save_interval=5)

    statistical_weights = np.ones_like(line_integrals)
    model_parameters = {
        'line_integrals': line_integrals,
        'statistical_weights': statistical_weights,
        'beta': beta,
        'projector': projector,
        'num_iters': number_of_iterations,
        'nonnegative': True,
        'objective_tolerance': objective_tolerance,
    }
    if optimizer_label == 'GD':
        model = PWLSReconstructionL2GD(**model_parameters)
    elif optimizer_label == 'CG':
        model = PWLSReconstructionL2CG(**model_parameters)
    elif optimizer_label == 'LBFGS':
        model = PWLSReconstructionL2LBFGS(**model_parameters)
    elif optimizer_label == 'FISTA':
        model = PWLSReconstructionL1FISTA(**model_parameters)
    else:
        raise ValueError(
            "optimizer_label must be 'GD', 'CG', 'LBFGS', or 'FISTA', "
            f"but got {optimizer_label!r}"
        )
    print(f'Using {optimizer_label} optimization.')

    if initial_image_option == 'zeros':
        initial_image = np.zeros(projector.image_shape, dtype=np.float64)
        print('Using an all-zero initial image.')
    else:
        if os.path.isfile(fbp_reconstruction_path):
            print(f'Loading FBP initial image: {fbp_reconstruction_path}')
            initial_image = fio.read_volume(fbp_reconstruction_path).squeeze().astype(np.float64)
        else:
            print('Saved FBP initial image not found; running FBFPD reconstruction.')
            fbp_reconstructor = FBFPD(
                projection_header=projection_header,
                projection=line_integrals,
                image_shape=phantom.shape,
                pixel_size=(phantom_header['drow'], phantom_header['dcol']),
                filter_type='hann',
            )
            initial_image = fbp_reconstructor.filtered_backprojection()
            rs.save_reconstruction_image(initial_image, fbp_reconstruction_path, phantom_header)
            print(f'Saved FBP initial image: {fbp_reconstruction_path}')

        initial_image = np.maximum(initial_image, 0.0)

    reconstruction = model.reconstruction(
        initial_image,
        verbose=True,
        callback=process_iteration,
    )
    rs.save_reconstruction_image(reconstruction, reconstruction_path, phantom_header)

rs.show_reconstruction_image(
    phantom,
    reconstruction,
    beta,
    number_of_iterations,
    reconstruction_title='PWLS reconstruction',
)
rs.convergence_analysis(iteration_save_folder)
rs.plot_matrices(iteration_save_folder, 'mae')
