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
from matplotlib.colors import to_rgb
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from tqdm import tqdm

from phase_only_orthonormalization.helper_functions import complex_to_rgb, complex_colorwheel
from phase_only_orthonormalization.mode_functions import trunc_gaussian
from phase_only_orthonormalization.directories import localdata


# ===== File settings ===== #
# Adjust this path to point to the location of the measurement data
path_glob = 'set12/wfs-comparison_t*.npz'           # Filename glob defining which files to read
file_numbers_to_include = list(range(0, 42))        # Which files to read and include in graph (at least two)
file_numbers_to_plot = []                 # From selection, which images to plot (non-existing are ignored)

do_plot_parking_convergence = True                  # Plot intermediate scans of auto-selecting an ROI around a bead

# ===== Plot settings ===== #
markers = ['.', 'x', '+']
colors = ['black', 'tab:blue', 'tab:green']

# ===== Image settings ===== #
cmap = 'magma'                                      # Colormap
slice0 = slice(0, 1000)                             # Crop x
slice1 = slice(0, 1000)                             # Crop y
circ_style = '--w'                                  # Pupil circle

# Scalebar
scalebar_props = {
    'width_m': 10e-6,
    'height_m': 700e-9,
    'pix_size_m': 27.5e-9,
    'pad': 0.4,
}

# Subplot settings
nrows = 2
ncols = 2

# Title strings
image_letters = ('a. ', 'b. ')
pattern_letters = ('c. ', 'd. ')
flat_wf_name = 'No correction'
basis_names = ('Plane wave basis', 'Plane wave Gaussian basis', 'Our orthonormal basis')

plt.rcParams['font.size'] = 14
colorwheel_fontsize = 17


# ===== Read files ===== #
assert len(file_numbers_to_include) > 1

# Use os.path.join to ensure the path is constructed correctly
full_path_glob = os.path.join(localdata, path_glob)
print(f'\nSearching for files matching {full_path_glob}')
npz_files_all = sorted(glob.glob(full_path_glob))
if len(npz_files_all) < len(file_numbers_to_include):
    raise ValueError(f'Selected {len(file_numbers_to_include)} files, but found only {len(npz_files_all)}, for glob: {full_path_glob}')
npz_files_sel = [npz_files_all[i] for i in file_numbers_to_include]

print(f'Found {len(npz_files_all)} files.')
print(f'Selected {len(file_numbers_to_include)} files.')

# Initialize
signal_improvements = [[], [], []]
contrast_enhancements = [[], [], []]
intermediate_results = [[], [], []]
snrs = [[], [], []]


# ===== Compute amplitude profile ===== #
# Compute initial coordinates and amplitude profile
domain = {
    'x_min': -1,
    'x_max': 1,
    'y_min': -1,
    'y_max': 1,
    'yxshape': (1000, 1000),           # Number of samples in each spatial direction
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

def draw_circle(circ_style, r_circ):
    theta_circ = np.linspace(0, 2 * np.pi, 250)
    x_circ = r_circ * np.cos(theta_circ)
    y_circ = r_circ * np.sin(theta_circ)
    plt.plot(x_circ, y_circ, circ_style)


def add_scalebar(ax, width_m, height_m, pix_size_m, pad):
    scalebar_width_pix = width_m / pix_size_m
    scalebar_height_pix = height_m / pix_size_m
    scalebar = AnchoredSizeBar(ax.transData,
                               scalebar_width_pix, f'{width_m*1e6:.0f}µm', 'lower left',
                               pad=pad,
                               color='white',
                               frameon=False,
                               size_vertical=scalebar_height_pix)

    ax.add_artist(scalebar)


for n_f, filepath in enumerate(tqdm(npz_files_sel)):

    # Load file
    npz_data = np.load(filepath, allow_pickle=True)

    # Algorithm index
    n_alg = npz_data['n_alg'][0]

    # Extract signal intensities and dark frame
    signal_before_flat = np.mean(npz_data['signal_before_flat'].squeeze())
    signal_after_flat = np.mean(npz_data['signal_after_flat'].squeeze())
    signal_shaped = np.mean(npz_data['signal_shaped'].squeeze())
    dark_frame = npz_data['dark_frame'][0]

    # Compute signal improvement and SNR
    signal_improvements[n_alg] += [(signal_shaped - dark_frame.mean()) / (signal_after_flat - dark_frame.mean())]
    snrs[n_alg] += [(signal_before_flat - dark_frame.mean()) / dark_frame.std()]

    # Contrast enhancements
    contrast_enhancements[n_alg] += [npz_data['contrast_results_all'][0]['contrast_enhancement']]

    # Intermediate results
    intermediate_results[n_alg] += [npz_data['wfs_result'][0].intermediate_results]

    # Plot images if requested
    if n_f in file_numbers_to_plot:
        # Initialize plot
        fig, axs = plt.subplots(nrows, ncols)
        fig.set_size_inches(10, 6)
        fig.set_dpi(120)
        plt.subplots_adjust(left=0.01, right=0.93, top=0.955, bottom=0.01, wspace=0.02, hspace=0.12)

        # Extract images
        img_shaped_wf = npz_data['contrast_results_all'][0]['img_shaped_wf']
        img_flat_wf = npz_data['contrast_results_all'][0]['img_flat_wf']
        vmin = 0
        vmax = np.maximum(np.percentile(img_shaped_wf, 100 - 1e-2), np.percentile(img_shaped_wf, 100 - 1e-2))

        # Beam parking info
        left, top, width, height = npz_data['park_result'][0]['location']
        xpark = left + width/2
        ypark = top + height/2

        im0 = plt.imshow(img_flat_wf[slice0, slice1], vmin=vmin, vmax=vmax, cmap=cmap)

        # Flat wavefront
        plt.subplot(nrows, ncols, 1)
        plt.title(f'{image_letters[0]}{flat_wf_name}')
        plt.xticks([])
        plt.yticks([])
        add_scalebar(ax=plt.gca(), **scalebar_props)

        # Show phase patterns
        scale = 1 / np.abs(amplitude).max()
        phase_rgb = complex_to_rgb(amplitude, scale)
        plt.subplot(nrows, ncols, ncols+1)
        plt.imshow(phase_rgb, extent=(-1, 1, -1, 1))
        plt.title(f'{pattern_letters[0]}')
        plt.xticks([])
        plt.yticks([])
        draw_circle(circ_style, 1)

        # Shaped wf
        plt.subplot(nrows, ncols, 2)
        img_shaped_wf = npz_data['contrast_results_all'][0]['img_shaped_wf']
        im1 = plt.imshow(img_shaped_wf[slice0, slice1], vmin=vmin, vmax=vmax, cmap=cmap)
        plt.title(f'{image_letters[1]}{basis_names[n_alg]}')
        plt.xticks([])
        plt.yticks([])
        add_scalebar(ax=plt.gca(), **scalebar_props)

        # Show phase patterns
        tc = npz_data['wfs_result'][0].t.conj()
        scale_tc = 1 / np.abs(tc).max()
        field_at_slm = amplitude * np.exp(1j * np.angle(tc))
        phase_rgb = complex_to_rgb(field_at_slm, 1)
        plt.subplot(nrows, ncols, ncols+2)
        plt.imshow(phase_rgb, extent=(-1, 1, -1, 1))
        plt.title(f'{pattern_letters[1]}')
        plt.xticks([])
        plt.yticks([])
        draw_circle(circ_style, 1)

    if n_f in file_numbers_to_plot:
        # Colorbar
        ax_cb = plt.axes((0.937, 0.516, 0.012, 0.433))
        fig.colorbar(im0, cax=ax_cb)
        ax_cb.set_ylabel('PMT signal')

        # Colorwheel
        ax_cw = plt.axes((0.84, 0.01, 0.10, 0.45))
        complex_colorwheel(ax=ax_cw, shape=(160, 160), text_kwargs={'fontsize': colorwheel_fontsize})

        # Plot parking convergence
        if do_plot_parking_convergence:
            fig2, axs2 = plt.subplots(2, 3)
            fig2.set_size_inches(14, 7)
            for i, img in enumerate(npz_data['park_result'][0]['imgs']):
                imshow = axs2.ravel()[i].imshow(img)
                fig2.colorbar(imshow, ax=axs2.ravel()[i])
                fig2.suptitle(f'{n_f}: Beam park convergence')


# Linear Least Squares
plt.figure()
# signal_improv_max = np.max([np.max(a) for a in signal_improvements])
# plt.plot((0, signal_improv_max), (0, signal_improv_max), '--', color='#999999', label='Equality')
for n_alg in range(len(signal_improvements)):
    plt.plot(snrs[n_alg], signal_improvements[n_alg], markers[n_alg], color=colors[n_alg], label=basis_names[n_alg])
plt.xlabel(f'SNR')
plt.ylabel(f'Signal improvement')
plt.title('Signal improvement')
plt.legend()

plt.figure()
for n_alg in range(len(contrast_enhancements)):
    plt.plot(snrs[n_alg], contrast_enhancements[n_alg], markers[n_alg], color=colors[n_alg], label=basis_names[n_alg])
plt.xlabel(f'SNR')
plt.ylabel(f'Contrast enhancement')
plt.title('Contrast enhancement')
plt.legend()


# Intermediate results - ping pong iterations
plt.figure()
for n_alg in range(len(intermediate_results)):
    dark = 0.3
    color = to_rgb(colors[n_alg])
    color_light = np.asarray(color) * dark + (1 - dark)
    plt.plot(np.transpose(intermediate_results[n_alg][0:16]), '.-', color=color_light)
    plt.plot(np.mean(intermediate_results[n_alg], axis=0), '.-', color=color, linewidth=2.5,
             label=f'{basis_names[n_alg]} mean')

plt.title('Ping pong convergence')
plt.xlabel('Ping pong index')
plt.ylabel('Signal')
plt.legend()
plt.ylim(bottom=0)

plt.show()

