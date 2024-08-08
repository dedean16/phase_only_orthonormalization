"""
Plot the orthonormalized plane wave basis and export a high-resolution version of the basis.
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py

from mode_functions import get_coords, coord_transform, trunc_gaussian, amplitude_rectangle
from helper_functions import plot_field, complex_colorwheel, get_dict_from_hdf5


# Settings
do_plot_bases = False
do_plot_jacobian = True
# filepath = 'C:/LocalData/ortho-plane-waves.hdf5'
filepath = '/home/dani/LocalData/ortho-plane-waves-1.hdf5'

# Jacobian plot
cmap = 'magma'

plt.rcParams['font.size'] = 12


# Import variables
with h5py.File(filepath, 'r') as f:
    init_modes = f['init_modes'][()]
    new_modes = f['new_modes'][()]
    a = f['a'][()]
    b = f['b'][()]
    p_tuple = f['p_tuple'][()]
    q_tuple = f['q_tuple'][()]
    domain = get_dict_from_hdf5(f['domain'])
    amplitude_kwargs = get_dict_from_hdf5(f['amplitude_kwargs'])


def draw_circle(circ_style, r_circ, theta_min, theta_max):
    theta_circ = np.linspace(theta_min, theta_max, 250)
    x_circ = r_circ * np.cos(theta_circ)
    y_circ = r_circ * np.sin(theta_circ)
    plt.plot(x_circ, y_circ, circ_style)


# Define number of rows and columns
n_rows = 5
n_cols_basis = 9
n_cols_total = 2 + 2 * n_cols_basis

subplot_index = (1 + np.flip(np.arange(n_rows * n_cols_basis).reshape((n_rows, n_cols_basis)), axis=0)
                 + (n_cols_basis + 2) * np.flip(np.expand_dims(np.arange(n_rows), axis=1), axis=0)).ravel()

# Plot end result
if do_plot_bases:
    scale = 1 / np.abs(init_modes[:, :, 0]).max()


    fig = plt.figure(figsize=(16, 8))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01)

    # Plot init functions
    for m, spi in enumerate(subplot_index):
        plt.subplot(n_rows, n_cols_total, spi)
        plot_field(init_modes[:, :, m], scale=scale)
        plt.xticks([])
        plt.yticks([])

    # Plot final functions
    for m, spi in enumerate(subplot_index):
        plt.subplot(n_rows, n_cols_total, spi+n_cols_basis+2)
        plot_field(new_modes[:, :, m], scale=scale)
        plt.xticks([])
        plt.yticks([])

    # Complex colorwheel
    center_spi = int(n_cols_basis + 1 + np.floor(n_rows/2) * n_cols_total)
    ax_cw = plt.subplot(n_rows, n_cols_total, (center_spi, center_spi+1))
    complex_colorwheel(ax=ax_cw, shape=(150, 150))

    # Title
    fig.text(0.23, 0.985, 'a. Initial functions', ha='center', va='center', fontsize=14)
    fig.text(0.77, 0.985, 'b. Our orthonormalized functions', ha='center', va='center', fontsize=14)


# === Jacobian === #
if do_plot_jacobian:
    # Normalization factor
    area = (domain['x_max'] - domain['x_min']) * (domain['y_max'] - domain['y_min'])
    num_samples = np.prod(domain['yxshape'])
    norm_factor = num_samples / area

    # Amplitude truncated Gaussian
    x, y = get_coords(domain)
    amp_unnorm_sq = trunc_gaussian(x, y, **amplitude_kwargs) ** 2       # Non-normalized amplitude squared
    amp_sq = amp_unnorm_sq * norm_factor / amp_unnorm_sq.sum()          # Normalized amplitude

    # Amplitude rectangle
    wx, wy, jacobian = coord_transform(x=x, y=y, a=a, b=b, p_tuple=p_tuple, q_tuple=q_tuple, compute_jacobian=True)
    amp_rect_sq_transformed = amplitude_rectangle(x=wx, y=wy, domain=domain) ** 2   # Transformed amplitude rectangle
    amp_sq_approx = norm_factor * (amp_rect_sq_transformed * jacobian.abs())        # Approx. to original amplitude²

    # Prepare plot
    # vmax = amp_sq.max()
    vmax = 1.2
    fig = plt.figure(figsize=(16, 8))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01)

    # Plot approximate amplitude functions
    for m, spi in enumerate(subplot_index):
        plt.subplot(n_rows, n_cols_total, spi + n_cols_basis + 2)
        plt.imshow(amp_sq_approx[:, :, m, 0, 0], vmin=0, vmax=vmax, cmap=cmap, extent=(-1, 0, -1, 1))
        draw_circle('--w', 1, np.pi/2, 3*np.pi/2)
        plt.xticks([])
        plt.yticks([])

    # Plot amplitude
    center_spi = int(n_cols_basis + 1 + np.floor(n_rows/2) * n_cols_total)
    plt.subplot(n_rows, n_cols_total, (center_spi, center_spi+1))
    plt.imshow(amp_sq[:, :, 0, 0, 0], vmin=0, vmax=vmax, cmap=cmap, extent=(-1, 0, -1, 1))
    draw_circle('--w', 1, np.pi / 2, 3 * np.pi / 2)
    plt.title('c. $A(x, y)$')
    plt.xticks([])
    plt.yticks([])

    ax_cb = plt.axes((0.490, 0.798, 0.012, 0.16))
    plt.colorbar(cax=ax_cb)
    plt.yticks((0, 0.5, 1.0, vmax), ('0.0', '0.5', '1.0', f'$\\geq${vmax:.1f}'))

    fig.text(0.23, 0.985, 'a. Transformed grids', ha='center', va='center', fontsize=14)
    fig.text(0.77, 0.985, "b. $A_R(x', y')\\cdot |\\,J\,|$", ha='center', va='center', fontsize=14)

    plt.show()
