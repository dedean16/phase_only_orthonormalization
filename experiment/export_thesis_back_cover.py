"""
This script is used to analyze the wavefront shaping measurements. Before running this script, please ensure that
directory paths in directories.py and the path matching variable path_glob defined below are valid. The variables
file_numbers_to_include and file_numbers_to_plot define which files must be included and plotted. Plotting all 35 files
at the same time takes up a significant amount of RAM.
"""
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
from skimage.io import imsave

from phase_only_orthonormalization.helper_functions import complex_to_rgb
from phase_only_orthonormalization.mode_functions import trunc_gaussian
from phase_only_orthonormalization.directories import localdata


export_thesis_back_filepath = os.path.join(localdata, 'thesis_back_cover.png')
export_tc_filepath = os.path.join(localdata, 'transmission_matrix.png')

# Adjust this path to point to the location of the measurement data
path_glob = 'wfs-comparison_t*.npz'                 # Filename glob defining which files to read
# file_numbers_to_include = list(range(0, 23))        # Which files to read and include in graph (at least two)
file_numbers_to_include = [16]        # Which files to read and include in graph (at least two)

do_plot_parking_convergence = False                 # Plot intermediate scans of auto-selecting an ROI around a bead
do_plot_extra_graphs = False

plt.rcParams['font.size'] = 14

assert len(file_numbers_to_include) > 0

# Use os.path.join to ensure the path is constructed correctly
full_path_glob = os.path.join(localdata, path_glob)
print(f'\nSearching for files matching {full_path_glob}')
npz_files_all = sorted(glob.glob(full_path_glob))
npz_files_sel = [npz_files_all[i] for i in file_numbers_to_include]

print(f'Found {len(npz_files_all)} files.')
print(f'Selected {len(file_numbers_to_include)} files.')


# Ny = 8000  # Number of pixels in vertical direction
Ny = 1000  # Number of pixels in vertical direction
AR = (170 + 5 + 7.7 / 2) / (240 + 2 * 5)  # Aspect Ratio
Nx = np.round(Ny * AR).astype(np.int32)

# ===== Compute amplitude profile ===== #
# Compute initial coordinates and amplitude profile
domain = {
    'x_min': -1.1*2*AR,
    'x_max': 0,
    'y_min': -1.1,
    'y_max': 1.1,
    'yxshape': (Ny, Nx),           # Number of samples in each spatial direction
}
x = np.linspace(domain['x_min'], domain['x_max'], domain['yxshape'][1]).reshape(1, -1)  # x coords
y = np.linspace(domain['y_min'], domain['y_max'], domain['yxshape'][0]).reshape(-1, 1)  # y coords

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


amplitude_kwargs = {'waist': waist, 'r_pupil': 1}
amplitude = trunc_gaussian(x, y, **amplitude_kwargs)
# ===================================== #

for n_f, filepath in enumerate(tqdm(npz_files_sel)):
    npz_data = np.load(filepath, allow_pickle=True)

    n_alg = 2   # Algorithm using orthonormal basis

    # Show phase patterns
    tc = npz_data['wfs_results_all'][0][n_alg].t.conj()
    scale_tc = 1 / np.abs(tc).max()

    Nx_input, Ny_input = tc.shape
    half_tc = tc[:, :Nx_input//2]
    half_tc_big_real = resize(half_tc.real, (Ny, Nx))
    half_tc_big_imag = resize(half_tc.imag, (Ny, Nx))
    field_at_slm = amplitude * np.exp(1j * np.angle(half_tc_big_real + 1j * half_tc_big_imag))

    slm_rgb = complex_to_rgb(field_at_slm, 1, colorspace='oklab')
    rgb_uint8 = np.round(255 * slm_rgb).astype(np.uint8)
    # imsave(export_thesis_back_filepath+f'{n_f}.png', rgb_uint8)
    imsave(export_thesis_back_filepath, rgb_uint8)

    rgb_tc = complex_to_rgb(tc, scale=scale_tc, colorspace='oklab')
    rgb_tc_uint8 = np.round(255 * rgb_tc).astype(np.uint8)
    # imsave(export_tc_filepath+f'{n_f}.png', rgb_tc_uint8)
    imsave(export_tc_filepath, rgb_tc_uint8)
