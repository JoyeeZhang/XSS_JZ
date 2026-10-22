import os
import matplotlib.pyplot as plt
import numpy as np
from XSSModules import file_io as fio
import matplotlib
matplotlib.use("TkAgg")

def add_poisson_noise(line_integrals, N0=1e5, water_lac=0.020585):
    absolute_line_integrals = line_integrals * water_lac
    expected_counts = N0 * np.exp(-absolute_line_integrals)
    noisy_counts = np.random.poisson(expected_counts)
    noisy_counts = np.maximum(noisy_counts, 1)
    noisy_line_integrals = -np.log(noisy_counts / N0) / water_lac

    return noisy_line_integrals


def save_reconstruction_image(image, output_path, reference_header):
    reconstruction_header = fio.create_header_dict_volume_basic(
        nslice=1,
        nrow=image.shape[0],
        ncol=image.shape[1],
        dslice=float(reference_header['dslice']),
        drow=float(reference_header['drow']),
        dcol=float(reference_header['dcol']),
        oslice=float(reference_header['oslice']),
        orow=float(reference_header['orow']),
        ocol=float(reference_header['ocol']),
        slice_thickness=float(reference_header['slice_thickness']),
        slope=1.0,
        intercept=0.0,
        data_type_numpy=np.float32,
    )
    fio.write_header(output_path, reconstruction_header)
    fio.write_volume(
        output_path,
        image[np.newaxis, :, :],
        dtype=np.float32,
    )


def process_iteration(iteration, image, objective, ground_truth, save_folder, save_interval=5):
    mae = np.mean(np.abs(image - ground_truth))
    print(f"Iteration {iteration}: MAE = {mae:.9e}")

    os.makedirs(save_folder, exist_ok=True)
    metrics_path = os.path.join(
        save_folder,
        f'PWLS_metrics.csv',
    )
    mode = 'w' if iteration == 0 else 'a'
    with open(metrics_path, mode) as file:
        if iteration == 0:
            file.write('iteration,mae,objective\n')
        file.write(f'{iteration},{mae:.12e},{objective:.12e}\n')

    if iteration > 0 and iteration % save_interval == 0:
        output_path = os.path.join(
            save_folder,
            f'PWLS_iteration{iteration:03d}_'
            f'{image.shape[1]}x{image.shape[0]}.png',
        )
        plt.imsave(
            output_path,
            image,
            cmap='gray',
            vmin=float(ground_truth.min()),
            vmax=float(ground_truth.max()),
        )
        print(f"Saved iteration {iteration}: {output_path}")


def plot_matrices(save_folder, metric):
    metric = str(metric).lower()
    metric_properties = {
        'mae': (1, 'MAE versus iteration', 'MAE [relative LAC]'),
        'objective': (2, 'Objective value versus iteration', 'Objective value'),
    }
    if metric not in metric_properties:
        raise ValueError("metric must be 'mae' or 'objective'")

    metrics_path = os.path.join(
        save_folder,
        f'PWLS_metrics.csv',
    )
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f'Metrics file not found: {metrics_path}')

    metrics = np.loadtxt(metrics_path, delimiter=',', skiprows=1)
    metrics = np.atleast_2d(metrics)
    if metrics.shape[1] != 3:
        raise ValueError(
            "metrics CSV must contain iteration, mae and objective columns"
        )

    value_column, title, ylabel = metric_properties[metric]
    iteration_numbers = metrics[:, 0].astype(np.int64)
    values = metrics[:, value_column]

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(iteration_numbers, values, marker='o', markersize=3)
    axis.set_title(title)
    axis.set_xlabel('Complete iteration')
    axis.set_ylabel(ylabel)
    axis.grid(True)
    figure.tight_layout()
    plt.show()


def show_reconstruction_image(
        ground_truth,
        reconstruction,
        beta=None,
        num_iterations=None,
        reconstruction_title='Reconstruction'):
    display_min = float(ground_truth.min())
    display_max = float(ground_truth.max())
    error_image = reconstruction - ground_truth
    rmse = np.sqrt(np.mean(error_image ** 2))
    mae = np.mean(np.abs(error_image))
    print(f"RMSE: {rmse:.9e}")
    print(f"MAE: {mae:.9e}")

    error_limit = max(
        float(np.max(np.abs(error_image))),
        np.finfo(float).eps,
    )

    figure = plt.figure(figsize=(16, 5), constrained_layout=True)
    grid = figure.add_gridspec(
        1,
        5,
        width_ratios=(1, 1, 0.05, 1, 0.05),
    )
    axes = (
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[0, 3]),
    )
    image_colorbar_axis = figure.add_subplot(grid[0, 2])
    error_colorbar_axis = figure.add_subplot(grid[0, 4])

    ground_truth_plot = axes[0].imshow(
        ground_truth,
        cmap='gray',
        vmin=display_min,
        vmax=display_max,
    )
    axes[0].set_title('Ground truth')
    axes[0].axis('image')
    axes[0].axis('off')

    axes[1].imshow(
        reconstruction,
        cmap='gray',
        vmin=display_min,
        vmax=display_max,
    )
    axes[1].set_title(reconstruction_title)
    axes[1].axis('image')
    axes[1].axis('off')

    figure.colorbar(
        ground_truth_plot,
        cax=image_colorbar_axis,
    )

    error_plot = axes[2].imshow(
        error_image,
        cmap='seismic',
        vmin=-error_limit,
        vmax=error_limit,
    )
    axes[2].set_title('Reconstruction minus ground truth')
    axes[2].axis('image')
    axes[2].axis('off')
    figure.colorbar(
        error_plot,
        cax=error_colorbar_axis,
    )

    if beta is not None and num_iterations is not None:
        figure.suptitle(f'PWLS: beta={beta:.1e}, {num_iterations} iterations')
    plt.show()


def convergence_analysis(save_folder):
    metrics_path = os.path.join(save_folder, 'PWLS_metrics.csv')
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f'Metrics file not found: {metrics_path}')

    metrics = np.loadtxt(metrics_path, delimiter=',', skiprows=1)
    metrics = np.atleast_2d(metrics)
    if metrics.shape[1] != 3:
        raise ValueError(
            'metrics CSV must contain iteration, mae and objective columns'
        )

    iteration_numbers = metrics[:, 0].astype(np.int64)
    errors = metrics[:, 1]
    log_errors = np.log(errors)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(iteration_numbers, log_errors, marker='o', markersize=3)
    axis.set_title('Convergence analysis')
    axis.set_xlabel('Iteration number')
    axis.set_ylabel(r'$\log(e_k)$')
    axis.grid(True)
    figure.tight_layout()
    plt.show()
