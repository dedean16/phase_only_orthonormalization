"""
Modify Zernike modes to be orthogonal in field.
"""
from itertools import count

import torch
import numpy as np
import matplotlib.pyplot as plt

from zernike import zernike_cart, zernike_order
from helper_functions import plot_field
from mode_functions import optimize_modes, apo_gaussian
from tilt_optim_functions import build_square_k_space


# ====== Settings ====== #
prefer_gpu = False  # Use cuda-GPU if it is available

if prefer_gpu and torch.cuda.is_available():
    torch.set_default_device('cuda')

do_plot = False
plot_per_its = 10  # Plot every this many iterations
do_save_plot = False
save_path_plot = 'C:/LocalData/mode_optimization_frames'
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

# Mode settings
k_max = 4

# Coefficients
poly_degree = 3         # Sqrt of number of polynomial terms
poly_per_mode = True    # If True, every mode has its own transform polynomial
pow_factor = 2

# Optimization parameters
learning_rate = 3.0e-2
iterations = 600
similarity_weight = 0.05
phase_grad_weight = 0.5


# ====== Initial basis ====== #
amplitude_kwargs = {'waist': waist, 'r_pupil': 1}

num_of_j = 8
# phase_coeffs = torch.tensor([np.pi]*num_of_j + [2*np.pi]*num_of_j)
phase_coeff_matrix = torch.cat((torch.eye(num_of_j) * np.pi, torch.eye(num_of_j) * 2 * np.pi), dim=0)
# phase_coeff_matrix = (torch.eye(num_of_j) * np.pi)
phase_coeff_matrix.requires_grad = True
phase_kwargs = {'phase_coeff_matrix': phase_coeff_matrix}
extra_params = {'phase_coeff_matrix': phase_coeff_matrix}


def zernike_phases(x, y, phase_coeff_matrix, dtype=torch.float32):
    """Compute the phases of a set of zernike modes."""
    phases = torch.zeros(y.shape[0], x.shape[1], phase_coeff_matrix.shape[0], 1, 1, dtype=dtype)
    for iM in range(phase_coeff_matrix.shape[0]):
        phase_coeffs = phase_coeff_matrix[iM, :]
        if x.shape[2] > 1:
            xmode = x[:, :, iM, 0, 0]
            ymode = y[:, :, iM, 0, 0]
        else:
            xmode = x[:, :, 0, 0, 0]
            ymode = y[:, :, 0, 0, 0]

        z_phase = 0
        for iJ, phase_coeff in enumerate(phase_coeffs):
            j = iJ + 2                                  # j=0 doesn't exist, and j=1 is piston
            n, m = zernike_order(j)
            z_phase += phase_coeff * zernike_cart(xmode, ymode, n, m)
        phases[:, :, iM, 0, 0] = z_phase
    return phases


# ====== Optimize modes ====== #
a, b, new_modes, init_modes = optimize_modes(
    domain=domain, amplitude_func=apo_gaussian, amplitude_kwargs=amplitude_kwargs, phase_func=zernike_phases,
    phase_kwargs=phase_kwargs, poly_degree=poly_degree, poly_per_mode=poly_per_mode, pow_factor=pow_factor,
    similarity_weight=similarity_weight, phase_grad_weight=phase_grad_weight, iterations=iterations,
    learning_rate=learning_rate, extra_params=extra_params)


print('\na:', a)
print('\nb:', b)


import matplotlib.pyplot as plt
from helper_functions import plot_field

nrows = phase_coeff_matrix.shape[0] // num_of_j
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
