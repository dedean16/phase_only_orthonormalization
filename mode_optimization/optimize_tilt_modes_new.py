import torch
import numpy as np

from mode_functions import optimize_modes, apo_gaussian
from tilt_optim_functions import build_square_k_space


# ====== Settings ====== #
prefer_gpu = False  # Use cuda-GPU if it is available

if prefer_gpu and torch.cuda.is_available():
    torch.set_default_device('cuda')

do_plot = False
plot_per_its = 25  # Plot every this many iterations
do_save_plot = False
save_path_plot = 'C:/LocalData/mode_optimization_frames'
save_path_coeffs = 'C:/LocalData'  # Where to save output

# Domain
domain = {
    'x_min': -1,
    'x_max': 0,
    'y_min': -1,
    'y_max': 1,
    'r_pupil': 1,
    'yxshape': (150, 75),           # Number of samples in each spatial direction
}

# Gaussian shape parameters
NA = 0.8  # Numerical Aperture
f_obj1_m = 12.5e-3  # Objective focal length in m
waist_m = 2 * 5.9e-3  # Fit beam profile gaussian width in m
waist = waist_m / (NA * f_obj1_m)

# Mode settings
k_max = 4

# Coefficients
poly_degree = 5  # Sqrt of number of polynomial terms
poly_per_mode = False
pow_factor = 1

# Optimization parameters
learning_rate = 3.0e-2
iterations = 700
similarity_weight = 0.01
phase_grad_weight = 0.01


# ====== Initial basis ====== #
amplitude_kwargs = {'waist': waist, 'r_pupil': 1}

kspace = torch.tensor(build_square_k_space(-k_max, k_max))
ky = kspace[0, :].view((1, 1, -1, 1, 1))
kx = kspace[1, :].view((1, 1, -1, 1, 1))
phase_kwargs = {'kx': kx, 'ky': ky}


def phase_gradient(x, y, kx, ky):
    return np.pi * (kx*x + ky*y)


# ====== Optimize modes ====== #
a, b, new_modes, init_modes = optimize_modes(
    domain=domain, amplitude_func=apo_gaussian, amplitude_kwargs=amplitude_kwargs, phase_func=phase_gradient,
    phase_kwargs=phase_kwargs, poly_degree=poly_degree, poly_per_mode=poly_per_mode, pow_factor=pow_factor,
    similarity_weight=similarity_weight, phase_grad_weight=phase_grad_weight, iterations=iterations,
    learning_rate=learning_rate, plot_per_its=plot_per_its)


print('\na:', a)
print('\nb:', b)

import matplotlib.pyplot as plt
from helper_functions import plot_field

scale = 50
plt.figure()
plot_field(init_modes[:,:,1].detach(), scale=scale)
plt.title('An old mode')

plt.figure()
plot_field(new_modes[:,:,1].detach(), scale=scale)
plt.title('A new mode')

plt.show()
