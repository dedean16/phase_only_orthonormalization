"""
Modify Zernike modes to be orthonormal in field, for a Gaussian amplitude profile. Before running this script, ensure
the paths in directories.py and the file paths defined in the save settings below are valid.
"""
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from zernike_functions import zernike_cart, zernike_order
from helper_functions import plot_field, complex_colorwheel
from mode_functions import optimize_modes, trunc_gaussian
from directories import localdata


# ====== Settings ====== #
do_plot = True
plot_per_its = 50  # Plot every this many iterations
do_plot_all_modes = True
do_plot_end = True
do_save_plot = False
save_path_plot = os.path.join(localdata, 'ortho-frames')                # Plot frames path
save_filename_plot = 'ortho_zernike_it'                                 # Plot frames filename prefix
save_path_coeffs = localdata
plt.rcParams['font.size'] = 12

# Domain
domain = {
    'x_min': -1,
    'x_max': 1,
    'y_min': -1,
    'y_max': 1,
    'yxshape': (100, 100),           # Number of samples in each spatial direction
}

# Gaussian shape parameters
NA = 0.8  # Numerical Aperture
f_obj1_m = 12.5e-3  # Objective focal length in m
pupil_radius_m = NA * f_obj1_m  # Radius of the objective pupil in meters
waist_radius_m = 2 * 5.9e-3  # Gaussian waist radius of amplitude profile on pupil plane in meters.
waist = waist_radius_m / pupil_radius_m  # Normalized waist radius for amplitude profile on pupil plane

# Note:
# (waist_radius_m for amplitude on pupil) = 2 * sqrt(2) * (waist_radius_m for intensity on SLM)
# Factor 2 is from SLM to pupil magnification (focal distances are 150mm and 300mm)
# Factor sqrt(2) is from intensity Gaussian profile to amplitude Gaussian profile

# Polynomial coefficients for transform
p_tuple = (0, 2, 4, 6, 8, 10)
q_tuple = (0, 2, 4, 6, 8, 10)
poly_per_mode = True    # If True, every mode has its own transform polynomial

# Optimization parameters
learning_rate = 1.5e-2
iterations = 8000
phase_grad_weight = 2.0


# ====== Initial basis ====== #
amplitude_kwargs = {'waist': waist, 'r_pupil': 1}

num_of_j = 10
phase_coeffs = torch.tensor([2*np.pi]*num_of_j + [-2*np.pi]*(num_of_j-1))
js = torch.tensor([*range(1, num_of_j+1), *range(2, num_of_j+1)])
phase_kwargs = {'phase_coeffs': phase_coeffs, 'js': js}
phase_coeffs.requires_grad = False
extra_params = {'phase_coeffs': phase_coeffs}

# Mode plotting
nrows = 3
ncols = num_of_j


def zernike_phases(x, y, phase_coeffs, js, dtype=torch.float32):
    """
    Compute the phases of a set of zernike modes.

    Args:
        x and y: 5D torch tensors containing the spatial coordinates. Dim 0 and 1 are for spatial coordinates. Dim 2 is
            for the mode index.
        phase_coeffs: Coefficient to multiply with the Zernike polynomial.
        js: Zernike polynomial indices.
        dtype: Data type.

    Returns:
        Torch tensor of Zernike polynomials.
    """
    phases = torch.zeros(y.shape[0], x.shape[1], phase_coeffs.shape[0], 1, 1, dtype=dtype)
    for i_mode in range(phase_coeffs.shape[0]):         # Loop over modes
        if x.shape[2] > 1:                          # Each mode has its own transformed coords
            x_mode = x[:, :, i_mode, 0, 0]
            y_mode = y[:, :, i_mode, 0, 0]
        else:                                       # One transform for all modes
            x_mode = x[:, :, 0, 0, 0]
            y_mode = y[:, :, 0, 0, 0]

        # Create Zernike mode with these coords
        phase_coeff = phase_coeffs[i_mode]
        j = js[i_mode]                                  # j=0 doesn't exist, and j=1 is piston
        n, m = zernike_order(j)
        phases[:, :, i_mode, 0, 0] = phase_coeff * zernike_cart(x_mode, y_mode, n, m)
    return phases


# ====== Optimize modes ====== #
a, b, new_modes, init_modes = optimize_modes(
    domain=domain, amplitude_func=trunc_gaussian, amplitude_kwargs=amplitude_kwargs, phase_func=zernike_phases,
    phase_kwargs=phase_kwargs, poly_per_mode=poly_per_mode, p_tuple=p_tuple, q_tuple=q_tuple,
    phase_grad_weight=phase_grad_weight, iterations=iterations,
    learning_rate=learning_rate, extra_params=extra_params, plot_per_its=plot_per_its, do_plot=do_plot, nrows=nrows,
    ncols=ncols, do_plot_all_modes=do_plot_all_modes, do_save_plot=do_save_plot, save_path_plot=save_path_plot)


print('\na:', a)
print('\nb:', b)
print('\nphase_coeffs:', phase_coeffs)

# Plot end result
if do_plot_end:
    # Prepare layout
    n_rows = 4
    n_cols_basis = 5                                                # Number of columns on one side
    n_cols_total = 1 + 2*n_cols_basis                               # Number of columns on whole subplot grid
    spi_skip = 10                                                   # Skip this subplot position

    # Prepare subplot indices
    subplot_index_half_grid = 1 + np.arange(n_rows * n_cols_basis).reshape((n_rows, n_cols_basis))
    subplot_index = np.delete(((subplot_index_half_grid
        + (n_cols_basis + 1) * np.expand_dims(np.arange(n_rows), axis=1)).ravel()), spi_skip)

    # Initialize figure with subplots
    fig = plt.figure(figsize=(16, 6.2))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.02, wspace=0.05, hspace=0.01)
    scale = 1 / np.abs(init_modes[:, :, 0]).max()

    # Plot init functions
    for m, spi in enumerate(subplot_index):
        plt.subplot(n_rows, n_cols_total, spi)
        plot_field(init_modes[:, :, m], scale=scale)
        plt.xticks([])
        plt.yticks([])

    # Plot final functions
    for m, spi in enumerate(subplot_index):
        plt.subplot(n_rows, n_cols_total, spi + n_cols_basis + 1)
        plot_field(new_modes[:, :, m].detach(), scale=scale)
        plt.xticks([])
        plt.yticks([])

    # Complex colorwheel
    center_spi = int(np.ceil(n_cols_total / 2))
    ax_cw = plt.subplot(1, n_cols_total, center_spi)
    complex_colorwheel(ax=ax_cw, shape=(150, 150))

    # Title
    fig.text(0.23, 0.98, 'a. Initial functions', ha='center', va='center', fontsize=14)
    fig.text(0.77, 0.98, 'b. Our orthonormalized functions', ha='center', va='center', fontsize=14)

    plt.show()
