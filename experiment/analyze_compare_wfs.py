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
from matplotlib.colors import TABLEAU_COLORS
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from tqdm import tqdm

from phase_only_orthonormalization.helper_functions import complex_to_rgb, complex_colorwheel, scalar_factor_least_squares
from phase_only_orthonormalization.mode_functions import trunc_gaussian
from phase_only_orthonormalization.directories import localdata


# Adjust this path to point to the location of the measurement data
path_glob = 'set14/wfs-comparison_t*.npz'                 # Filename glob defining which files to read
file_numbers_to_include = list(range(0, 20))        # Which files to read and include in graph (at least two)
file_numbers_to_plot = []                 # From selection, which images to plot (non-existing are ignored)

do_plot_parking_convergence = True                  # Plot intermediate scans of auto-selecting an ROI around a bead

# Image settings
cmap = 'magma'                                      # Colormap
slice0 = slice(0, 1000)                              # Crop x
slice1 = slice(0, 1000)                             # Crop y
circ_style = '--w'                                  # Pupil circle

flat_beadline_x = (544, 613)                        # Line through bead x coords, flat wavefront
flat_beadline_y = (257, 257)                        # Line through bead y coords, flat wavefront
algs_beadline_x = ((534, 603), (542, 611), (1,3)), (1,3)          # Line through bead x coords, shaped wavefront algorithms
algs_beadline_y = ((245, 245), (255, 255), (1,3)), (1,3)          # Line through bead y coords, shaped wavefront algorithms
beadline_flat_kwargs = {'color': (0.0, 0.9, 1.0), 'linewidth': 0.5}
beadline_alg_kwargs = ({'color': (1.0, 0.7, 0.0), 'linewidth': 0.5}, {'color': (0.1, 1.0, 0.1), 'linewidth': 0.5}, {}, {})

bead_flat_kwargs = {'color': 'tab:blue', 'linestyle': 'dashdot'}
bead_alg0_kwargs = {'color': 'tab:orange', 'linestyle': 'dashed'}
bead_alg1_kwargs = {'color': 'tab:green', 'linestyle': 'solid'}

# Scalebar
scalebar_props = {
    'width_m': 10e-6,
    'height_m': 700e-9,
    'pix_size_m': 27.5e-9,
    'pad': 0.4,
}

# Subplot settings
nrows = 2
ncols = 5

# Title strings
# image_letters = ('a. ', 'b. ', 'c. ')
# pattern_letters = ('e. ', 'f. ', 'g. ')
image_letters = ('a. ', 'b. ', 'c. ', 'x')
pattern_letters = ('e. ', 'f. ', 'g. ', 'x')
beadline_letter = 'd.'
basis_names = ('No correction', 'Alg index 0', 'Alg index 1', 'Alg index 2', 'Alg index 3')

plt.rcParams['font.size'] = 14
colorwheel_fontsize = 17

assert len(file_numbers_to_include) > 1

# Use os.path.join to ensure the path is constructed correctly
full_path_glob = os.path.join(localdata, path_glob)
print(f'\nSearching for files matching {full_path_glob}')
npz_files_all = sorted(glob.glob(full_path_glob))
npz_files_sel = [npz_files_all[i] for i in file_numbers_to_include]

print(f'Found {len(npz_files_all)} files.')
print(f'Selected {len(file_numbers_to_include)} files.')

# Initialize
signal_enhancement = [[], [], [], []]
fidelity_noise = [[], [], [], []]
signal_before_flat_all = []
signal_after_flat_all = []
signal_shaped_all = []
intermediate_results = []
signals_before_flat_photobleach = []
signals_after_flat_photobleach = []


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


def add_beadline(ax, beadline_x, beadline_y, beadline_kwargs):
    ax.plot(beadline_x, beadline_y, **beadline_kwargs)


def get_data_beadline(img, beadline_x, beadline_y, pix_avg_plusmin=3):
    slice_y = slice(beadline_y[0]-pix_avg_plusmin, beadline_y[1]+pix_avg_plusmin)
    slice_x = slice(beadline_x[0], beadline_x[1]+1)
    return img[slice_y, slice_x].mean(axis=0)


for n_f, filepath in enumerate(tqdm(npz_files_sel)):

    intermediate_results.append([])

    # Load file
    npz_data = np.load(filepath, allow_pickle=True)

    if n_f in file_numbers_to_plot:
        # Initialize plot
        fig, axs = plt.subplots(nrows, ncols)
        fig.set_size_inches(14.2, 6)
        fig.set_dpi(120)
        plt.subplots_adjust(left=0.01, right=0.93, top=0.955, bottom=0.01, wspace=0.02, hspace=0.12)

    # Extract images
    img_shaped_wf = npz_data['contrast_results_all'][0][1]['img_shaped_wf']
    img_flat_wf = npz_data['contrast_results_all'][0][1]['img_flat_wf']
    vmin = 0
    vmax = np.maximum(np.percentile(img_shaped_wf, 100 - 1e-2), np.percentile(img_shaped_wf, 100 - 1e-2))

    # Beam parking info
    left, top, width, height = npz_data['park_result'][0]['location']
    xpark = left + width/2
    ypark = top + height/2

    signals_before_flat_photobleach += [np.mean(npz_data['signal_before_flat'], axis=(2, 3)).squeeze()]
    signals_after_flat_photobleach += [np.mean(npz_data['signal_after_flat'], axis=(2, 3)).squeeze()]

    # Flat wavefront
    if n_f in file_numbers_to_plot:
        img_flat_wf = npz_data['contrast_results_all'][0][0]['img_flat_wf']
        plt.subplot(nrows, ncols, 1)
        im0 = plt.imshow(img_flat_wf[slice0, slice1], vmin=vmin, vmax=vmax, cmap=cmap)
        plt.title(f'{image_letters[0]}{basis_names[0]}')
        plt.xticks([])
        plt.yticks([])
        add_beadline(plt.gca(), flat_beadline_x, flat_beadline_y, beadline_flat_kwargs)
        add_scalebar(ax=plt.gca(), **scalebar_props)

        # Show phase patterns
        scale = 1 / np.abs(amplitude).max()
        phase_rgb = complex_to_rgb(amplitude, scale)
        plt.subplot(nrows, ncols, ncols+1)
        plt.gca().set_position((0.172, 0.01, 0.141, 0.446))
        plt.imshow(phase_rgb, extent=(-1, 1, -1, 1))
        plt.title(f'{pattern_letters[0]}')
        plt.xticks([])
        plt.yticks([])
        draw_circle(circ_style, 1)

    for n_alg in range(len(npz_data['algorithm_types'])):
        intermediate_results[n_f].append([])
        alg_str = npz_data['algorithm_types'][n_alg]

        if n_f in file_numbers_to_plot:
            # Load images
            img_shaped_wf = npz_data['contrast_results_all'][0][n_alg]['img_shaped_wf']

            # Shaped wf
            plt.subplot(nrows, ncols, 2+n_alg)
            im1 = plt.imshow(img_shaped_wf[slice0, slice1], vmin=vmin, vmax=vmax, cmap=cmap)
            plt.title(f'{image_letters[1+n_alg]}{basis_names[1+n_alg]}')
            plt.xticks([])
            plt.yticks([])
            add_beadline(plt.gca(), algs_beadline_x[n_alg], algs_beadline_y[n_alg], beadline_alg_kwargs[n_alg])
            add_scalebar(ax=plt.gca(), **scalebar_props)

            # Show phase patterns
            tc = npz_data['wfs_results_all'][0][n_alg].t.conj()
            scale_tc = 1 / np.abs(tc).max()
            field_at_slm = amplitude * np.exp(1j * np.angle(tc))
            phase_rgb = complex_to_rgb(field_at_slm, 1)
            plt.subplot(nrows, ncols, 2+ncols+n_alg)
            if n_alg == 0:
                plt.gca().set_position((0.407, 0.01, 0.141, 0.446))
            if n_alg == 1:
                plt.gca().set_position((0.627, 0.01, 0.141, 0.446))
            plt.imshow(phase_rgb, extent=(-1, 1, -1, 1))
            plt.title(f'{pattern_letters[1 + n_alg]}')
            plt.xticks([])
            plt.yticks([])
            draw_circle(circ_style, 1)

        # Extract signal intensities
        signal_before_flat_all += [np.mean(npz_data['signal_before_flat'].squeeze())]
        signal_after_flat_all += [np.mean(npz_data['signal_after_flat'].squeeze())]
        signal_shaped_all += [np.mean(npz_data['signal_shaped'].squeeze())]

        signal_enhancement[n_alg] += \
            [np.mean(npz_data['signal_shaped'].squeeze()[n_alg]) / np.mean(npz_data['signal_after_flat'].squeeze()[n_alg])]

        fidelity_noise[n_alg] += [npz_data['wfs_results_all'][0, n_alg].results_all[0].fidelity_noise[0]]

        intermediate_results[n_f][n_alg] += [*npz_data['wfs_results_all'][0, n_alg].intermediate_results]

    if n_f in file_numbers_to_plot:
        # Colorbar
        ax_cb = plt.axes((0.937, 0.516, 0.012, 0.433))
        fig.colorbar(im0, cax=ax_cb)
        ax_cb.set_ylabel('PMT signal')

        # Colorwheel
        ax_cw = plt.axes((0.84, 0.01, 0.10, 0.45))
        complex_colorwheel(ax=ax_cw, shape=(160, 160), text_kwargs={'fontsize': colorwheel_fontsize})

        # Signal intensity line through bead
        plt.axes((0.060, 0.120, 0.080, 0.282))

        img_shaped_wf0 = npz_data['contrast_results_all'][0][0]['img_shaped_wf']
        img_shaped_wf1 = npz_data['contrast_results_all'][0][1]['img_shaped_wf']
        beadsignal_flat = get_data_beadline(img_flat_wf, flat_beadline_x, flat_beadline_y).squeeze()
        beadsignal_alg0 = get_data_beadline(img_shaped_wf0, algs_beadline_x[0], algs_beadline_y[0]).squeeze()
        beadsignal_alg1 = get_data_beadline(img_shaped_wf1, algs_beadline_x[1], algs_beadline_y[1]).squeeze()
        num_pix_beadline = (flat_beadline_x[1] - flat_beadline_x[0] + 1)
        xrange = np.linspace(0, scalebar_props['pix_size_m'] * 1e6 * num_pix_beadline, num_pix_beadline)

        plt.plot(xrange, beadsignal_flat, label=basis_names[0], **bead_flat_kwargs)
        plt.plot(xrange, beadsignal_alg0, label=basis_names[1], **bead_alg0_kwargs)
        plt.plot(xrange, beadsignal_alg1, label=basis_names[2], **bead_alg1_kwargs)
        plt.xlabel('position ($\\mu m$)')
        plt.ylabel('PMT signal through bead')
        plt.title(beadline_letter)

        # Plot parking convergence
        if do_plot_parking_convergence:
            fig2, axs2 = plt.subplots(2, 3)
            fig2.set_size_inches(14, 7)
            for i, img in enumerate(npz_data['park_result'][0]['imgs']):
                imshow = axs2.ravel()[i].imshow(img)
                fig2.colorbar(imshow, ax=axs2.ravel()[i])
                fig2.suptitle(f'{n_f}: Beam park convergence')



# Linear Least Squares
# improvement_ratio_1 = scalar_factor_least_squares(signal_enhancement[0], signal_enhancement[1])[0]
# improvement_ratio_2 = scalar_factor_least_squares(signal_enhancement[0], signal_enhancement[2])[0]
# improvement_ratio_3 = scalar_factor_least_squares(signal_enhancement[0], signal_enhancement[3])[0]
#
# improvement_ratio_12 = scalar_factor_least_squares(signal_enhancement[1], signal_enhancement[2])[0]
# improvement_ratio_13 = scalar_factor_least_squares(signal_enhancement[1], signal_enhancement[3])[0]
# improvement_ratio_23 = scalar_factor_least_squares(signal_enhancement[2], signal_enhancement[3])[0]

mean_signal_enhancement = np.mean(signal_enhancement, axis=1)
print(f'Average signal improvement factor {npz_data["algorithm_types"][0]}: {mean_signal_enhancement[0]:.4f}')
print(f'Average signal improvement factor {npz_data["algorithm_types"][1]}: {mean_signal_enhancement[1]:.4f}')
print(f'Average signal improvement factor {npz_data["algorithm_types"][2]}: {mean_signal_enhancement[2]:.4f}')
# print(f'Average signal improvement factor ratio (least squares): {improvement_ratio_1:.4f}')
# print(f'Average signal improvement factor ratio (least squares): {improvement_ratio_2:.4f}')

plt.figure()
n_algs = range(len(signal_enhancement))
plt.plot(np.asarray(fidelity_noise).T.squeeze(), np.asarray(signal_enhancement).T, '.',
         label=[f'{basis_names[n+1]}' for n in n_algs])
plt.xlabel('Fidelity noise of first ping pong iteration')
plt.ylabel('Signal enhancement')
plt.legend()

plt.figure()
signal_enh_max = np.max(signal_enhancement)
plt.plot((0, signal_enh_max), (0, signal_enh_max), '--', color='#999999', label='Equality')
# plt.plot((0, signal_enh_max / improvement_ratio_1), (0, signal_enh_max), color='tab:green', label='Least squares fit')
plt.plot(signal_enhancement[0], signal_enhancement[1], '.', color='tab:blue', label='Signal improvement')
plt.xlabel(f'{basis_names[1]}')
plt.ylabel(f'{basis_names[2]}')
plt.title('Signal improvement factor')
plt.legend()

plt.figure()
signal_enh_max = np.max(signal_enhancement)
plt.plot((0, signal_enh_max), (0, signal_enh_max), '--', color='#999999', label='Equality')
# plt.plot((0, signal_enh_max / improvement_ratio_2), (0, signal_enh_max), color='tab:green', label='Least squares fit')
plt.plot(signal_enhancement[0], signal_enhancement[2], '.', color='tab:blue', label='Signal improvement')
plt.xlabel(f'{basis_names[1]}')
plt.ylabel(f'{basis_names[3]}')
plt.title('Signal improvement factor')
plt.legend()

plt.figure()
signal_enh_max = np.max(signal_enhancement)
plt.plot((0, signal_enh_max), (0, signal_enh_max), '--', color='#999999', label='Equality')
# plt.plot((0, signal_enh_max / improvement_ratio_3), (0, signal_enh_max), color='tab:green', label='Least squares fit')
plt.plot(signal_enhancement[0], signal_enhancement[3], '.', color='tab:blue', label='Signal improvement')
plt.xlabel(f'{basis_names[1]}')
plt.ylabel(f'{basis_names[4]}')
plt.title('Signal improvement factor')
plt.legend()

plt.figure()
signal_enh_max = np.max(signal_enhancement)
plt.plot((0, signal_enh_max), (0, signal_enh_max), '--', color='#999999', label='Equality')
# plt.plot((0, signal_enh_max / improvement_ratio_12), (0, signal_enh_max), color='tab:green', label='Least squares fit')
plt.plot(signal_enhancement[1], signal_enhancement[2], '.', color='tab:blue', label='Signal improvement')
plt.xlabel(f'{basis_names[2]}')
plt.ylabel(f'{basis_names[3]}')
plt.title('Signal improvement factor')
plt.legend()

plt.figure()
signal_enh_max = np.max(signal_enhancement)
plt.plot((0, signal_enh_max), (0, signal_enh_max), '--', color='#999999', label='Equality')
# plt.plot((0, signal_enh_max / improvement_ratio_13), (0, signal_enh_max), color='tab:green', label='Least squares fit')
plt.plot(signal_enhancement[1], signal_enhancement[3], '.', color='tab:blue', label='Signal improvement')
plt.xlabel(f'{basis_names[2]}')
plt.ylabel(f'{basis_names[4]}')
plt.title('Signal improvement factor')
plt.legend()

plt.figure()
signal_enh_max = np.max(signal_enhancement)
plt.plot((0, signal_enh_max), (0, signal_enh_max), '--', color='#999999', label='Equality')
# plt.plot((0, signal_enh_max / improvement_ratio_23), (0, signal_enh_max), color='tab:green', label='Least squares fit')
plt.plot(signal_enhancement[2], signal_enhancement[3], '.', color='tab:blue', label='Signal improvement')
plt.xlabel(f'{basis_names[3]}')
plt.ylabel(f'{basis_names[4]}')
plt.title('Signal improvement factor')
plt.legend()

plt.figure()
plt.plot(np.asarray(signal_enhancement), '.-', color='gray')
plt.plot(np.asarray(signal_enhancement).mean(axis=1), color='red', linewidth=2.5, label='Mean')
plt.xlabel('Algorithm index')
plt.ylabel('Signal improvement')
plt.title('Signal improvement vs alg index')
plt.ylim((0, None))
plt.legend()

plt.figure()
plt.plot(np.asarray(signals_before_flat_photobleach).T, '.-', color='gray')
plt.plot(np.asarray(signals_before_flat_photobleach).T.mean(axis=1), color='red', linewidth=2.5, label='Mean')
plt.xlabel('Algorithm index')
plt.ylabel('Signal')
plt.title('Photobleaching\nflat wf signal measurements before alg')
plt.ylim((0, None))
plt.legend()

plt.figure()
plt.plot(np.asarray(signals_after_flat_photobleach).T, '.-', color='gray')
plt.plot(np.asarray(signals_after_flat_photobleach).T.mean(axis=1), color='red', linewidth=2.5, label='Mean')
plt.xlabel('Algorithm index')
plt.ylabel('Signal')
plt.title('Photobleaching\nflat wf signal measurements after alg')
plt.ylim((0, None))
plt.legend()

plt.figure()
plt.plot(signal_before_flat_all, signal_shaped_all, '.')
plt.xlabel('Signal flat (before running alg)')
plt.ylabel('Signal shaped')
plt.title('Signal flat vs signal shaped')

plt.figure()
plt.plot(np.asarray(intermediate_results)[:, 0, :].squeeze().T, '.-', color='#99ccff')
plt.plot(np.asarray(intermediate_results)[:, 1, :].squeeze().T, '.-', color='#ffaaaa')
plt.plot(np.asarray(intermediate_results)[:, 2, :].squeeze().T, '.-', color='#aaffaa')
plt.plot(np.asarray(intermediate_results)[:, 3, :].squeeze().T, '.-', color='#eeeeaa')

plt.plot(np.asarray(intermediate_results)[:, 0, :].squeeze().T.mean(axis=1), '.-', color='#3355ff', linewidth=2.5, label=f'{basis_names[1]} mean')
plt.plot(np.asarray(intermediate_results)[:, 1, :].squeeze().T.mean(axis=1), '.-', color='#ff2222', linewidth=2.5, label=f'{basis_names[2]} mean')
plt.plot(np.asarray(intermediate_results)[:, 2, :].squeeze().T.mean(axis=1), '.-', color='#22ff22', linewidth=2.5, label=f'{basis_names[3]} mean')
plt.plot(np.asarray(intermediate_results)[:, 3, :].squeeze().T.mean(axis=1), '.-', color='#eeee22', linewidth=2.5, label=f'{basis_names[4]} mean')

plt.title('Ping pong convergence')
plt.xlabel('Ping pong index')
plt.ylabel('Signal')
plt.legend()
plt.ylim(bottom=0)

plt.show()

