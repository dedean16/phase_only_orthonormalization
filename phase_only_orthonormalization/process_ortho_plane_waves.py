"""
Plot the orthonormalized plane wave basis and export a high-resolution version of the basis.
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py

from mode_functions import get_coords, coord_transform, trunc_gaussian
from helper_functions import plot_field, complex_colorwheel, get_dict_from_hdf5


# Settings
do_plot_functions = True
filepath = 'C:/LocalData/ortho-plane-waves.hdf5'


with h5py.File(filepath, 'r') as f:
    init_modes = f['init_modes'][()]
    new_modes = f['new_modes'][()]
    a = f['a'][()]
    b = f['b'][()]
    p_tuple = f['p_tuple'][()]
    q_tuple = f['q_tuple'][()]
    domain = get_dict_from_hdf5(f['domain'])
    amplitude_kwargs = get_dict_from_hdf5(f['amplitude_kwargs'])


# Plot end result
if do_plot_functions:
    n_rows = 5
    n_cols_basis = 9
    n_cols_total = 2 + 2*n_cols_basis
    scale = 1 / np.abs(init_modes[:, :, 0]).max()

    subplot_index = (1 + np.flip(np.arange(n_rows * n_cols_basis).reshape((n_rows, n_cols_basis)), axis=0)
                     + (n_cols_basis+2) * np.flip(np.expand_dims(np.arange(n_rows), axis=1), axis=0)).ravel()

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
    x, y = get_coords(domain)
    wx, wy, jacobian = coord_transform(x=x, y=y, a=a, b=b, p_tuple=p_tuple, q_tuple=q_tuple, compute_jacobian=True)
    amplitude_unnorm = trunc_gaussian(x, y, **amplitude_kwargs)

    plt.figure()
    plt.imshow(jacobian[:, :, 0, 0, 0].abs(), vmin=0, vmax=10)
    plt.title('$|J|$')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.figure()
    plt.imshow(amplitude_unnorm[:, :, 0, 0, 0], vmin=0, vmax=1)
    plt.title('$A/A_0$')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.figure()
    plt.imshow((amplitude_unnorm*jacobian.abs())[:, :, 0, 0, 0], vmin=0, vmax=3)
    plt.title('$A|J|/A_0$')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.show()
