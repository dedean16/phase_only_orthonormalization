"""
Orthonormalize a phase-only version of Laguerre Gaussian modes, for a Gaussian amplitude profile. Before running this
script, ensure the paths in directories.py and the file paths defined in the save settings below are valid.
"""
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py

from mode_functions import optimize_modes, trunc_gaussian, laguerre_gauss_phase_factor
from helper_functions import add_dict_as_hdf5group, gitinfo, plot_field, complex_colorwheel
from directories import localdata


# ====== Settings ====== #
do_plot = True
plot_per_its = 50  # Plot every this many iterations
do_plot_end = True
do_save_plot = False
do_save_result = True
do_plot_all_modes = True
save_path_plot = os.path.join(localdata, 'ortho-frames')            # Plot frames path
save_filename_plot = 'ortho-lg-it'                                  # Plot frames filename prefix
save_filepath_result = os.path.join(localdata, 'ortho-lg.hdf5')     # Where to save output
plt.rcParams['font.size'] = 12

# Note: Figures saved as images can be turned into a video with ffmpeg:
# e.g.: ffmpeg -i ortho-lg-it%04d.png -framerate 60 -c:v libx265 -pix_fmt yuv420p -crf 20 ortho-plane-waves.mp4

# Domain
domain = {
    'x_min': -1,
    'x_max': 1,
    'y_min': -1,
    'y_max': 1,
    'yxshape': (150, 150),           # Number of samples in each spatial direction
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
p_tuple = (0, 2, 4)
q_tuple = (0, 2, 4)
poly_per_mode = True    # If True, every mode has its own transform polynomial

# Optimization parameters
learning_rate = 1.0e-4
iterations = 8000
phase_grad_weight = 0.01     # 1/w²


# ====== Initial basis ====== #
amplitude_kwargs = {'waist': waist, 'r_pupil': 1}
phase_kwargs = {'el_max': 3, 'p_max': 2, 'w0': waist/3, 'step_smoothness': 0.01}

# Mode plotting
nrows = 4
ncols = 7



# ====== Optimize modes ====== #
a, b, new_modes, init_modes = optimize_modes(
    domain=domain, amplitude_func=trunc_gaussian, amplitude_kwargs=amplitude_kwargs, phase_func=None,
    phase_factor_func=laguerre_gauss_phase_factor, compute_phase_gradient=True,
    phase_kwargs=phase_kwargs, poly_per_mode=poly_per_mode, p_tuple=p_tuple, q_tuple=q_tuple,
    phase_grad_weight=phase_grad_weight, iterations=iterations,
    learning_rate=learning_rate, plot_per_its=plot_per_its, do_save_plot=do_save_plot, do_plot=do_plot,
    save_path_plot=save_path_plot, save_filename_plot=save_filename_plot, ncols=ncols, nrows=nrows,
    do_plot_all_modes=do_plot_all_modes)


print('\na:', a)
print('\nb:', b)


if do_plot_end:
    # Prepare layout
    n_rows = 7
    n_cols_basis = 3                                                # Number of columns on one side
    n_cols_total = 1 + 2*n_cols_basis                               # Number of columns on whole subplot grid

    # Prepare subplot indices
    subplot_index = 1 + np.arange(n_rows * n_cols_total).reshape((n_cols_total, n_rows))[:, 0:n_cols_basis].T.ravel()

    # Initialize figure with subplots
    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(left=0.005, right=0.995, top=0.965, bottom=0.005, wspace=0.04, hspace=0.005)
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
    fig.text(0.78, 0.98, 'b. Our orthonormalized functions', ha='center', va='center', fontsize=14)

    plt.show()

# Save result
if do_save_result:
    with h5py.File(save_filepath_result, 'w') as f:
        # Coefficients and modes
        f.create_dataset('a', data=a.detach().numpy())
        f.create_dataset('b', data=b.detach().numpy())
        f.create_dataset('new_modes', data=new_modes.detach().numpy())
        f.create_dataset('init_modes', data=init_modes.detach().numpy())

        # Parameters
        f.create_dataset('p_tuple', data=p_tuple)
        f.create_dataset('q_tuple', data=q_tuple)
        f.create_dataset('poly_per_mode', data=poly_per_mode)
        f.create_dataset('learning_rate', data=learning_rate)
        f.create_dataset('phase_grad_weight', data=phase_grad_weight)
        f.create_dataset('iterations', data=iterations)
        f.create_dataset('amplitude_func_name', data=trunc_gaussian.__name__)
        f.create_dataset('phase_func_name', data='')
        f.create_dataset('phase_factor_func_name', data=laguerre_gauss_phase_factor.__name__)

        # Dictionaries
        add_dict_as_hdf5group(name='domain', dic=domain, hdf=f)
        add_dict_as_hdf5group(name='amplitude_kwargs', dic=amplitude_kwargs, hdf=f)
        add_dict_as_hdf5group(name='phase_kwargs', dic=phase_kwargs, hdf=f)
        add_dict_as_hdf5group(name='git_info', dic=gitinfo(), hdf=f)
