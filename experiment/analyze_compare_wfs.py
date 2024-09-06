import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from tqdm import tqdm

from phase_only_orthonormalization.helper_functions import complex_to_rgb, complex_colorwheel, scalar_factor_least_squares
from phase_only_orthonormalization.mode_functions import trunc_gaussian


# Adjust this path to point to the location of the measurement data
localdata = '/home/dani/LocalData/wfs-OrthoFBDR-comparison/'
# localdata = 'C:/LocalData/wfs-wfsr-comparison/'
path_glob = 'set?/wfs-comparison_t*.npz'
file_numbers_to_plot = list(range(0, 5))        # Which images to plot

do_plot_parking_convergence = False

# Image settings
cmap = 'magma'
slice0 = slice(0, 1000)
slice1 = slice(0, 1000)
circ_style = '--w'

# Scalebar
scalebar_props = {
    'width_m': 10e-6,
    'height_m': 700e-9,
    'pix_size_m': 40e-9,
    'pad': 0.4,
}

# Subplot settings
nrows = 2
ncols = 3

# Title strings
image_letters = ('a. ', 'b. ', 'c. ')
pattern_letters = ('d. ', 'e. ', 'f. ')
basis_names = ('No correction', 'Plane wave basis', 'Our orthonormal basis')

plt.rcParams['font.size'] = 13


# Use os.path.join to ensure the path is constructed correctly
npz_files = sorted(glob.glob(os.path.join(localdata, path_glob)))

print(f'Found {len(npz_files)} files.')

# Initialize
signal_enhancement = [[], []]
signal_flat_all = []
signal_shaped_all = []
full_pattern_feedback_all = []


# ===== Compute amplitude profile ===== #
# Compute initial coordinates and amplitude profile
domain = {
    'x_min': -1,
    'x_max': 1,
    'y_min': -1,
    'y_max': 1,
    'yxshape': (300, 300),           # Number of samples in each spatial direction
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


for n_f, filepath in enumerate(tqdm(npz_files)):

    full_pattern_feedback_all.append([])

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

    # Flat wavefront
    if n_f in file_numbers_to_plot:
        img_flat_wf = npz_data['contrast_results_all'][0][0]['img_flat_wf']
        plt.subplot(nrows, ncols, 1)
        im0 = plt.imshow(img_flat_wf[slice0, slice1], vmin=vmin, vmax=vmax, cmap=cmap)
        plt.title(f'{image_letters[0]}{basis_names[0]}')
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

    for n_alg in range(2):
        full_pattern_feedback_all[n_f].append([])
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
            add_scalebar(ax=plt.gca(), **scalebar_props)

            # Show phase patterns
            tc = npz_data['wfs_results_all'][0][n_alg].t.conj()
            scale = 1 / np.abs(tc).max()
            phase_rgb = complex_to_rgb(tc, scale)
            plt.subplot(nrows, ncols, 2+ncols+n_alg)
            plt.imshow(phase_rgb, extent=(-1, 1, -1, 1))
            plt.title(f'{pattern_letters[1 + n_alg]}')
            plt.xticks([])
            plt.yticks([])
            draw_circle(circ_style, 1)

        # Extract signal intensities
        signal_flat_all += [np.mean(npz_data['signal_flat'].squeeze())]
        signal_shaped_all += [np.mean(npz_data['signal_shaped'].squeeze())]

        signal_enhancement[n_alg] += \
            [np.mean(npz_data['signal_shaped'].squeeze()[n_alg]) / np.mean(npz_data['signal_flat'].squeeze()[n_alg])]

        full_pattern_feedback_all[n_f][n_alg] += [*npz_data['wfs_results_all'][0, n_alg].full_pattern_feedback]

    if n_f in file_numbers_to_plot:
        # Colorbar
        ax_cb = plt.axes((0.937, 0.506, 0.014, 0.451))
        fig.colorbar(im0, cax=ax_cb)
        ax_cb.set_ylabel('Signal')

        # Colorwheel
        ax_cw = plt.axes((0.91, 0.01, 0.08, 0.45))
        complex_colorwheel(ax=ax_cw, shape=(160, 160))

        # Plot parking convergence
        if do_plot_parking_convergence:
            fig2, axs2 = plt.subplots(2, 3)
            fig2.set_size_inches(14, 7)
            for i, img in enumerate(npz_data['park_result'][0]['imgs']):
                imshow = axs2.ravel()[i].imshow(img)
                fig2.colorbar(imshow, ax=axs2.ravel()[i])
                fig2.suptitle(f'{n_f}: Beam park convergence')



# Linear Least Squares
improvement_ratio = scalar_factor_least_squares(signal_enhancement[0], signal_enhancement[1])[0]

mean_signal_enhancement = np.mean(signal_enhancement, axis=1)
print(f'Average signal improvement factor {npz_data["algorithm_types"][0]}: {mean_signal_enhancement[0]:.4f}')
print(f'Average signal improvement factor {npz_data["algorithm_types"][1]}: {mean_signal_enhancement[1]:.4f}')
print(f'Average signal improvement factor ratio (least squares): {improvement_ratio:.4f}')

plt.figure()
signal_enh_max = np.max(signal_enhancement)
plt.plot((0, signal_enh_max), (0, signal_enh_max), '--', color='#999999', label='Equality')
plt.plot((0, signal_enh_max / improvement_ratio), (0, signal_enh_max), color='tab:green', label='Least squares fit')
plt.plot(signal_enhancement[0], signal_enhancement[1], '.', color='tab:blue', label='Signal improvement')
plt.xlabel(f'{basis_names[1]}')
plt.ylabel(f'{basis_names[2]}')
plt.title('Signal improvement factor')
plt.legend(loc=4)

plt.figure()
plt.plot(signal_flat_all, signal_shaped_all, '.')
plt.xlabel('Signal flat')
plt.ylabel('Signal shaped')
plt.title('Signal flat vs signal shaped')

plt.figure()
plt.plot(np.asarray(full_pattern_feedback_all)[:, 0, :].squeeze().T, '.-', color='#99ccff')
plt.plot(np.asarray(full_pattern_feedback_all)[:, 1, :].squeeze().T, '.-', color='#ffaaaa')

plt.plot(np.asarray(full_pattern_feedback_all)[:, 0, :].squeeze().T.mean(axis=1), '.-', color='#3355ff', linewidth=2.5, label=f'{npz_data["algorithm_types"][0]} mean')
plt.plot(np.asarray(full_pattern_feedback_all)[:, 1, :].squeeze().T.mean(axis=1), '.-', color='#ff2222', linewidth=2.5, label=f'{npz_data["algorithm_types"][1]} mean')

plt.title('Ping pong convergence')
plt.xlabel('Ping pong index')
plt.ylabel('Signal')
plt.legend()
plt.ylim(bottom=0)

plt.show()

