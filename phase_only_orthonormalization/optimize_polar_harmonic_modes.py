import torch
import numpy as np
import matplotlib.pyplot as plt

from mode_functions import optimize_modes, trunc_gaussian
from helper_functions import plot_field


# ====== Settings ====== #
prefer_gpu = False  # Use cuda-GPU if it is available

if prefer_gpu and torch.cuda.is_available():
    torch.set_default_device('cuda')

do_plot = True
plot_per_its = 30  # Plot every this many iterations
do_save_plot = False
do_plot_all_modes = True
do_plot_end = True
save_path_plot = 'C:/LocalData/mode_optimization_frames_tilt'
save_filename_plot = 'mode_optimization_it'
save_path_coeffs = 'C:/LocalData'  # Where to save output

# Note: Figures saved as images can be turned into a video with ffmpeg:
# e.g.: ffmpeg -i mode_optimization_it%04d.png -framerate 60 -c:v libx265 -pix_fmt yuv420p -crf 20 mode_optimization.mp4

# Domain
domain = {
    'x_min': -1,
    'x_max': 0,
    'y_min': -1,
    'y_max': 1,
    'r_pupil': 1,
    'yxshape': (200, 100),           # Number of samples in each spatial direction
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
# Gaussian shape parameters

# Polynomial coefficients for transform
p_tuple = (0, 2, 4, 6)
q_tuple = (0, 2, 4, 6)
poly_per_mode = False    # If True, every mode has its own transform polynomial

# Optimization parameters
learning_rate = 2.5e-2
iterations = 1001
phase_grad_weight = 0.0


# ====== Initial basis ====== #
def phase_gradient(x, y, k_r, k_t):
    r_sq = x * x + y * y
    theta = torch.angle(x + 1j * y)

    # Compute modes
    k_rb, k_tb = torch.broadcast_tensors(k_r, k_t)
    k_R = k_rb.reshape(1, 1, -1, 1, 1)
    k_T = k_tb.reshape(1, 1, -1, 1, 1)
    phi = (k_R * (2 * np.pi) * r_sq + k_T * theta)
    return phi


amplitude_kwargs = {'waist': waist, 'r_pupil': 1}

k_r = torch.tensor(np.arange(-2, 3).reshape(-1, 1))
k_t = torch.tensor(np.arange(-4, 5, 2).reshape(1, -1))
phase_kwargs = {'k_r': k_r, 'k_t': k_t}

# Mode plotting
nrows = 6
ncols = 5



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

# Plot end result
if do_plot_end:
    nrows = 5
    ncols = 5
    scale = 60

    plt.figure(figsize=(16, 6), dpi=80)
    plt.tight_layout()
    plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
    plt.suptitle('Initial modes')

    # Loop over modes
    for i in range(init_modes.shape[2]):
        plt.subplot(nrows, ncols, i+1)
        plot_field(init_modes[:, :, i], scale=scale)
        plt.xticks([])
        plt.yticks([])


    plt.figure(figsize=(16, 6), dpi=80)
    plt.tight_layout()
    plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
    plt.suptitle('New modes')

    for i in range(new_modes.shape[2]):
        plt.subplot(nrows, ncols, i+1)
        plot_field(new_modes[:, :, i].detach(), scale=scale)
        plt.xticks([])
        plt.yticks([])

    plt.show()
