"""
Modify Zernike modes to be orthogonal in field.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from zernike_functions import zernike_cart, zernike_order
from helper_functions import plot_field
from mode_functions import optimize_modes, apo_gaussian


# ====== Settings ====== #
prefer_gpu = True  # Use cuda-GPU if it is available

if prefer_gpu and torch.cuda.is_available():
    torch.set_default_device('cuda')

do_plot = True
plot_per_its = 1  # Plot every this many iterations
do_plot_all_modes = True
do_plot_end = True
do_save_plot = True
save_path_plot = 'C:/LocalData/mode_optimization_frames'   ### TODO: check if directory exists
save_filename_plot = 'mode_optimization_zernike_it'
save_path_coeffs = 'C:/LocalData'  # Where to save output

# Domain
domain = {
    'x_min': -1,
    'x_max': 1,
    'y_min': -1,
    'y_max': 1,
    'r_pupil': 1,
    'yxshape': (100, 100),           # Number of samples in each spatial direction
}

# Gaussian shape parameters
NA = 0.8  # Numerical Aperture
f_obj1_m = 12.5e-3  # Objective focal length in m
waist_m = 2 * 5.9e-3  # Fit beam profile gaussian width in m
waist = waist_m / (NA * f_obj1_m)

# Coefficients
poly_degree = 8         # Sqrt of number of polynomial terms
poly_per_mode = True    # If True, every mode has its own transform polynomial
pow_factor = 1

# Optimization parameters
learning_rate = 2.5e-2
iterations = 1000
similarity_weight = 0.02
phase_grad_weight = 0.2


# ====== Initial basis ====== #
amplitude_kwargs = {'waist': waist, 'r_pupil': 1}

num_of_j = 8
phase_coeffs = torch.tensor([np.pi]*num_of_j + [2*np.pi]*num_of_j)
js = torch.tensor(list(range(2, num_of_j+2)) * 2)
phase_kwargs = {'phase_coeffs': phase_coeffs, 'js': js}
phase_coeffs.requires_grad = False
extra_params = {}

# Mode plotting
nrows = 1 + phase_coeffs.shape[0] // num_of_j
ncols = num_of_j


def zernike_phases(x, y, phase_coeffs, js, dtype=torch.float32):
    """
    Compute the phases of a set of zernike modes.

    Args:
        x and y:
        phase_coeffs:
        js:
        dtype:

    Returns:

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
    domain=domain, amplitude_func=apo_gaussian, amplitude_kwargs=amplitude_kwargs, phase_func=zernike_phases,
    phase_kwargs=phase_kwargs, poly_degree=poly_degree, poly_per_mode=poly_per_mode, pow_factor=pow_factor,
    similarity_weight=similarity_weight, phase_grad_weight=phase_grad_weight, iterations=iterations,
    learning_rate=learning_rate, extra_params=extra_params, plot_per_its=plot_per_its, do_plot=do_plot, nrows=nrows,
    ncols=ncols, do_plot_all_modes=do_plot_all_modes, do_save_plot=do_save_plot, save_path_plot=save_path_plot)


print('\na:', a)
print('\nb:', b)

# Plot end result
if do_plot_end:
    nrows = phase_coeffs.shape[0] // num_of_j
    ncols = num_of_j
    scale = 50

    plt.figure(figsize=(15, 8), dpi=80)
    plt.tight_layout()
    plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
    plt.suptitle('Initial modes')

    # Loop over modes
    for i in range(init_modes.shape[2]):
        plt.subplot(nrows, ncols, i+1)
        plot_field(init_modes[:, :, i], scale=scale)
        plt.xticks([])
        plt.yticks([])


    plt.figure(figsize=(15, 8), dpi=80)
    plt.tight_layout()
    plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
    plt.suptitle('New modes')

    for i in range(new_modes.shape[2]):
        plt.subplot(nrows, ncols, i+1)
        plot_field(new_modes[:, :, i].detach(), scale=scale)
        plt.xticks([])
        plt.yticks([])

    plt.show()
