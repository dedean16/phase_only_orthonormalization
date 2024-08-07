import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py

from mode_functions import optimize_modes, trunc_gaussian
from helper_functions import add_dict_as_hdf5group, gitinfo


# ====== Settings ====== #
prefer_gpu = False  # Use cuda-GPU if it is available

if prefer_gpu and torch.cuda.is_available():
    torch.set_default_device('cuda')

do_plot = True
plot_per_its = 50  # Plot every this many iterations
do_save_plot = False
do_save_result = False
do_plot_all_modes = True
save_path_plot = 'C:/LocalData/ortho-plane-waves-frames'
save_filename_plot = 'ortho-plane-waves-it'
save_filepath_result = 'C:/LocalData/ortho-plane-waves.hdf5'  # Where to save output
plt.rcParams['font.size'] = 12

# Note: Figures saved as images can be turned into a video with ffmpeg:
# e.g.: ffmpeg -i ortho-plane-waves-it%04d.png -framerate 60 -c:v libx265 -pix_fmt yuv420p -crf 20 ortho-plane-waves.mp4

# Domain
domain = {
    'x_min': -1,
    'x_max': 0,
    'y_min': -1,
    'y_max': 1,
    'yxshape': (150, 75),           # Number of samples in each spatial direction
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

# Mode settings
k_max = 4

# Polynomial coefficients for transform
p_tuple = (0, 1, 2, 3, 4, 5, 6, 7)
q_tuple = (0, 2, 4, 6, 8, 10)
poly_per_mode = True    # If True, every mode has its own transform polynomial

# Optimization parameters
learning_rate = 1.0e-2
iterations = 8001
# iterations = 10
phase_grad_weight = 0.2


# ====== Initial basis ====== #
def build_square_k_space(k_min, k_max):
    """
    Constructs the k-space by creating a set of (k_y, k_x) coordinates.
    Fills the k_left and k_right matrices with the same k-space. (k_y, k_x) denote the k-space coordinates of the whole
    pupil. Only half SLM (and thus pupil) is modulated at a time, hence k_x (axis=1) must make steps of 2.

    Returns:
        k_space (np.ndarray): A 2xN array of k-space coordinates.
    """
    # Generate kx and ky coordinates
    ky_angles = np.arange(k_min, k_max + 1, 1)
    k_angles_min_even = (k_min if k_min % 2 == 0 else k_min + 1)        # Must be even
    kx_angles = np.arange(k_angles_min_even, k_max + 1, 2)              # Steps of 2

    # Combine ky and kx coordinates into pairs
    k_y = np.repeat(np.array(ky_angles)[np.newaxis, :], len(kx_angles), axis=0).flatten()
    k_x = np.repeat(np.array(kx_angles)[:, np.newaxis], len(ky_angles), axis=1).flatten()
    k_space = np.vstack((k_y, k_x))
    return k_space


def phase_gradient(x, y, kx, ky):
    return np.pi * (kx*x + ky*y)


amplitude_kwargs = {'waist': waist, 'r_pupil': 1}

kspace = torch.tensor(build_square_k_space(-k_max, k_max))
ky = kspace[0, :].view((1, 1, -1, 1, 1))
kx = kspace[1, :].view((1, 1, -1, 1, 1))
phase_kwargs = {'kx': kx, 'ky': ky}

# Mode plotting
nrows = 4
ncols = 15



# ====== Optimize modes ====== #
a, b, new_modes, init_modes = optimize_modes(
    domain=domain, amplitude_func=trunc_gaussian, amplitude_kwargs=amplitude_kwargs, phase_func=phase_gradient,
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
        f.create_dataset('phase_func_name', data=phase_gradient.__name__)

        # Dictionaries
        add_dict_as_hdf5group(name='domain', dic=domain, hdf=f)
        add_dict_as_hdf5group(name='amplitude_kwargs', dic=amplitude_kwargs, hdf=f)
        add_dict_as_hdf5group(name='phase_kwargs', dic=phase_kwargs, hdf=f)
        add_dict_as_hdf5group(name='git_info', dic=gitinfo(), hdf=f)
