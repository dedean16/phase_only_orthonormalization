"""
Orthonormalize a phase-only version of Laguerre Gaussian modes.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py

from mode_functions import optimize_modes, trunc_gaussian, laguerre_gauss_phase_factor
from helper_functions import add_dict_as_hdf5group, gitinfo


# ====== Settings ====== #
prefer_gpu = False  # Use cuda-GPU if it is available

if prefer_gpu and torch.cuda.is_available():
    torch.set_default_device('cuda')

do_plot = True
plot_per_its = 50  # Plot every this many iterations
do_save_plot = False
do_save_result = True
do_plot_all_modes = True
save_path_plot = 'C:/LocalData/ortho-lg-frames'
save_filename_plot = 'ortho-lg-it'
save_filepath_result = 'C:/LocalData/ortho-lg.hdf5'  # Where to save output
plt.rcParams['font.size'] = 12

# Note: Figures saved as images can be turned into a video with ffmpeg:
# e.g.: ffmpeg -i ortho-plane-waves-it%04d.png -framerate 60 -c:v libx265 -pix_fmt yuv420p -crf 20 ortho-plane-waves.mp4

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
# iterations = 10
phase_grad_weight = 0.4     # 1/w²


# ====== Initial basis ====== #
amplitude_kwargs = {'waist': waist, 'r_pupil': 1}
phase_kwargs = {'el_max': 2, 'p_max': 2, 'w0': waist/2, 'step_smoothness': 0.05}

# Mode plotting
nrows = 4
ncols = 5



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
