"""
Plot the orthonormalized plane wave basis and the jacobians and export a high-resolution version of the basis. Before
running this script, ensure the paths in directories.py and the file paths defined in the settings below are valid.
"""
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from skimage.transform import resize

from mode_functions import get_coords, coord_transform, trunc_gaussian, amplitude_rectangle, compute_modes, \
    phase_gradient
from helper_functions import plot_field, complex_colorwheel, get_dict_from_hdf5, add_dict_as_hdf5group, gitinfo, \
    grid_bitmap
from directories import localdata


# Settings
do_plot_bases = True
do_plot_transform_jacobian = True
do_plot_transformed_gridmap = False
do_export_modes = True

import_filepath = os.path.join(localdata, 'ortho-plane-waves.hdf5')
export_filepath = os.path.join(localdata, 'ortho-plane-waves-hires.hdf5')

# Transformed grids
num_grid_cells = 6
arc_linewidth = 1.8
grid_linewidth = 0.8
arccolor = 'tab:blue'
arcstyle = '--'
gridstyle = '-k'

# Jacobian plot
cmap = 'magma'

plt.rcParams['font.size'] = 12


# Import variables
with h5py.File(import_filepath, 'r') as f:
    # Coefficients and modes
    a = f['a'][()]
    b = f['b'][()]
    init_modes = f['init_modes'][()]
    new_modes = f['new_modes'][()]

    # Parameters
    p_tuple = f['p_tuple'][()]
    q_tuple = f['q_tuple'][()]
    poly_per_mode = f['poly_per_mode'][()]
    learning_rate = f['learning_rate'][()]
    phase_grad_weight = f['phase_grad_weight'][()]
    iterations = f['iterations'][()]

    # Dictionaries
    domain = get_dict_from_hdf5(f['domain'])
    amplitude_kwargs = get_dict_from_hdf5(f['amplitude_kwargs'])
    phase_kwargs = get_dict_from_hdf5(f['phase_kwargs'])
    git_info_orthonormalization = get_dict_from_hdf5(f['git_info'])

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

    fig = plt.figure(figsize=(14, 7))
    plt.subplots_adjust(left=0.005, right=0.995, top=0.965, bottom=0.005, wspace=0.03, hspace=0.05)

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

    # Axis domain
    domain_spi = int(n_cols_basis - 1 + np.floor(n_rows/2))
    ax_domain = plt.subplot(n_rows, n_cols_total, (domain_spi, domain_spi+1))
    ax_domain.set_xlim((-1, 0))
    ax_domain.set_ylim((-1, 1))
    ax_domain.set_xlabel('x', labelpad=-2.0)
    ax_domain.set_ylabel('y', labelpad=-3.0, rotation=0)
    ax_domain.set_aspect(1)
    ax_domain.set_title('Domain')

    # Title
    fig.text(0.23, 0.985, 'a. Initial functions', ha='center', va='center', fontsize=14)
    fig.text(0.77, 0.985, 'b. Our orthonormalized functions', ha='center', va='center', fontsize=14)


# Plot transformed grid and Jacobian
if do_plot_transform_jacobian:
    # Prepare plot
    fig = plt.figure(figsize=(12, 6))
    plt.subplots_adjust(left=0.005, right=0.995, top=0.922, bottom=0.005, hspace=0.045, wspace=0.02)

    # === Plot transformed grid === #
    gridsize_x = 1 + num_grid_cells * (domain['x_max'] - domain['x_min'])
    gridsize_y = 1 + num_grid_cells * (domain['y_max'] - domain['y_min'])
    x_grid_lin = torch.linspace(domain['x_min'], domain['x_max'], gridsize_x)
    x_grid = x_grid_lin.view(1, -1, 1, 1, 1).broadcast_to((gridsize_y, gridsize_x, 1, 1, 1))
    y_grid_lin = torch.linspace(domain['y_min'], domain['y_max'], gridsize_y)
    y_grid = y_grid_lin.view(-1, 1, 1, 1, 1).broadcast_to((gridsize_y, gridsize_x, 1, 1, 1))
    r_mask = x_grid * x_grid + y_grid * y_grid > 1.01

    wx_grid, wy_grid = coord_transform(x_grid, y_grid, a, b, p_tuple, q_tuple)
    r_mask_bc = r_mask.broadcast_to(wx_grid.shape)
    wx_grid[r_mask_bc] = np.nan
    wy_grid[r_mask_bc] = np.nan

    # Warped arc
    phi_arc = torch.linspace(np.pi / 2, 3 * np.pi / 2, 60)
    x_arc = torch.cos(phi_arc).view(-1, 1, 1, 1, 1)
    y_arc = torch.sin(phi_arc).view(-1, 1, 1, 1, 1)
    wx_arc, wy_arc = coord_transform(x_arc, y_arc, a, b, p_tuple, q_tuple)

    for m, spi in enumerate(subplot_index):
        plt.subplot(n_rows, n_cols_total, spi)
        plt.plot(wx_arc[:, :, m, 0, 0], wy_arc[:, :, m, 0, 0], arcstyle, color=arccolor, linewidth=arc_linewidth)
        plt.plot(wx_grid[:, :, m, 0, 0], wy_grid[:, :, m, 0, 0], gridstyle, linewidth=grid_linewidth)
        plt.plot(wx_grid[:, :, m, 0, 0].T, wy_grid[:, :, m, 0, 0].T, gridstyle, linewidth=grid_linewidth)
        plt.xlim((domain['x_min'], domain['x_max']))
        plt.ylim((domain['y_min'], domain['y_max']))
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_aspect(1)

    # Plot amplitude
    middletop_spi = n_cols_basis + 1
    ax_middletop = plt.subplot(n_rows, n_cols_total, (middletop_spi, middletop_spi + 1))
    ax_middletop.set_aspect(1)
    x_grid_disk = x_grid.clone()
    y_grid_disk = y_grid.clone()
    x_grid_disk[r_mask] = np.nan
    y_grid_disk[r_mask] = np.nan
    plt.plot(x_arc[:, :, 0, 0, 0], y_arc[:, :, 0, 0, 0], arcstyle, color=arccolor, linewidth=arc_linewidth)
    plt.plot(x_grid_disk[:, :, 0, 0, 0], y_grid_disk[:, :, 0, 0, 0], gridstyle, linewidth=grid_linewidth)
    plt.plot(x_grid_disk[:, :, 0, 0, 0].T, y_grid_disk[:, :, 0, 0, 0].T, gridstyle, linewidth=grid_linewidth)
    plt.title('b. Arc and\ngrid (x, y)')
    plt.xlim((domain['x_min'], domain['x_max']))
    plt.ylim((domain['y_min'], domain['y_max']))
    plt.xticks([])
    plt.yticks([])

    # === Plot approximate amplitude functions, obtained from Jacobians === #
    # Normalization factor
    area = (domain['x_max'] - domain['x_min']) * (domain['y_max'] - domain['y_min'])
    num_samples = np.prod(domain['yxshape'])
    norm_factor = num_samples / area

    # Amplitude truncated Gaussian
    x_hr, y_hr = get_coords(domain)
    amp_unnorm_sq = trunc_gaussian(x_hr, y_hr, **amplitude_kwargs) ** 2       # Non-normalized amplitude squared
    amp_sq = amp_unnorm_sq * norm_factor / amp_unnorm_sq.sum()          # Normalized amplitude

    # Amplitude rectangle
    wx, wy, jacobian = coord_transform(x=x_hr, y=y_hr, a=a, b=b, p_tuple=p_tuple, q_tuple=q_tuple, compute_jacobian=True)
    amp_rect_sq_transformed = amplitude_rectangle(x=wx, y=wy, domain=domain) ** 2   # Transformed amplitude rectangle
    amp_sq_approx = norm_factor * (amp_rect_sq_transformed * jacobian.abs())        # Approx. to original amplitude²

    # vmax = amp_sq.max()
    vmax = 1.2
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
    plt.title('d. $A^2(x, y)$')
    plt.xticks([])
    plt.yticks([])

    ax_cb = plt.axes((0.480, 0.05, 0.01, 0.28))
    plt.colorbar(cax=ax_cb)
    plt.yticks((0, 0.5, 1.0, vmax), ('0.0', '0.5', '1.0', f'$\\geq${vmax:.1f}'))

    fig.text(0.23, 0.96, "a. Transformed arcs and grids (x', y')", ha='center', va='center', fontsize=14)
    fig.text(0.77, 0.96, "c. $A_R^2(x', y')\\cdot |\\,J\,|$", ha='center', va='center', fontsize=14)


# Plot a transformed bitmap image of a grid
if do_plot_transformed_gridmap:
    fig = plt.figure(figsize=(16, 8))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01)

    domain_bitmap = {**domain, 'yxshape': (2000, 1000)}
    x_bm, y_bm = get_coords(domain_bitmap)

    for m, spi in enumerate(subplot_index):
        am = a[:, :, m:m+1, :, :]
        bm = b[:, :, m:m+1, :, :]
        wx_bm, wy_bm = coord_transform(x=x_bm, y=y_bm, a=am, b=bm, p_tuple=p_tuple, q_tuple=q_tuple)
        gridmap_hr = grid_bitmap(wx_bm[:, :, 0, 0, 0], wy_bm[:, :, 0, 0, 0], domain_bitmap, 0.1, 0.002)
        gridmap = np.clip(50 * resize(gridmap_hr.numpy().astype(np.float32), (200, 100)), a_min=0.0, a_max=1.0)
        plt.subplot(n_rows, n_cols_total, spi)
        plt.imshow(gridmap, cmap='gray')
        plt.xticks([])
        plt.yticks([])

plt.show()


if do_export_modes:
    print('Generating hi-res modes...')
    domain_hr = {**domain, 'yxshape': (1000, 500)}
    x_hr, y_hr = get_coords(domain_hr)
    amplitude_profile = trunc_gaussian(x_hr, y_hr, **amplitude_kwargs)
    amplitude = torch.tensor(1.0)
    init_phases_hr = np.angle(compute_modes(amplitude, phase_gradient, phase_kwargs, x_hr, y_hr)[0])

    wx, wy = coord_transform(x_hr, y_hr, a, b, p_tuple, q_tuple)
    new_phases_hr = np.angle(compute_modes(amplitude, phase_gradient, phase_kwargs, wx, wy)[0])

    with h5py.File(export_filepath, 'w') as f:
        # Coefficients and modes
        f.create_dataset('a', data=a)
        f.create_dataset('b', data=b)
        f.create_dataset('new_phases_hr', data=new_phases_hr)
        f.create_dataset('init_phases_hr', data=init_phases_hr)
        f.create_dataset('amplitude_profile', data=amplitude_profile)

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
        add_dict_as_hdf5group(name='git_info_orthonormalization', dic=git_info_orthonormalization, hdf=f)

        print(f'Exported hi-res modes ({domain_hr["yxshape"][0]}x{domain_hr["yxshape"][1]}) to {export_filepath}')
