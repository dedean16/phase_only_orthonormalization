from typing import Tuple, Dict

import torch
from torch import Tensor as tt
from torch.optim import Optimizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from helper_functions import plot_field


def inner(A: tt, B: tt, dim):
    """
    Inner product.

    Args:
        A, B: Tensors containields to compute inner product for.
        dim: Tensor dimension to take the inner product over.

    Returns: Inner product (Surprise, surprise!)
    """
    return (A * B.conj()).sum(dim=dim)


def apo_gaussian(x: tt, y: tt, waist, r_pupil):
    """
    Compute an apodized Gaussian on the given x & y coordinates.

    Args:
        x: Tensor containing the x spatial coordinates.
        y: Tensor containing the y spatial coordinates.
        waist: Gaussian waist.
        r_pupil: Pupil radius for apodization.
    
    Returns: Values for apodized Gaussian beam profile.
    """
    r_sq = x ** 2 + y ** 2
    return torch.exp(-(r_sq / waist**2)) * (r_sq <= r_pupil)


def coord_transform(x: tt, y: tt, a: tt, b: tt, poly_powers_x: Tuple[int, ...], poly_powers_y: Tuple[int, ...]):
    """
    Bivariate polynomial coordinate transform.

    Dim 0 and 1 are used for spatial coordinate index. Dim 2 for mode index. Dim 3 and 4 for polynomial power index.

    Args:
        x: 5D Tensor containing the x spatial coordinates.
        y: 5D Tensor containing the y spatial coordinates.
        a: 5D Tensor containing the x polynomial coefficients
        b: 5D Tensor containing the y polynomial coefficients
        poly_powers_x: polynomial powers for x
        poly_powers_y: polynomial powers for y

    Returns:
        wx, wy: The transformed (warped) coordinates x' and y'.
    """
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)
    assert a.shape == b.shape

    # Create arrays of powers
    xpows = torch.tensor(poly_powers_x).view(1, 1, 1, -1, 1)
    ypows = torch.tensor(poly_powers_y).view(1, 1, 1, 1, -1)

    # Raise x and y to even powers
    x_to_powers = x ** xpows
    y_to_powers = y ** ypows

    # Multiply all factors and sum to create the polynomial
    wx = (a * x * x_to_powers * y_to_powers).sum(dim=(3, 4), keepdim=True)
    wy = (b * y * x_to_powers * y_to_powers).sum(dim=(3, 4), keepdim=True)

    return wx, wy


def compute_gram(modes: tt) -> tt:
    """
    Compute Gram matrix from collection of modes.

    Args:
        modes: a 3D tensor containing all 2D modes. Dim 0 and 1 are used for spatial coordinates,
            and dim 2 is the mode index.

    Returns:
        gram: Gram matrix
    """
    num_modes = modes.shape[2]
    gram = torch.zeros(num_modes, num_modes, dtype=modes.dtype)

    for n in range(num_modes):
        row_mode = modes[:, :, n:n+1]
        gram[:, n] = inner(modes, row_mode, dim=(0, 1))

    return gram


def compute_non_orthonormality(modes: tt) -> Tuple[tt, tt]:
    """
    Compute non-orthonormality

    Compute the non-orthonormality of a set of modes. For orthonormal bases, this value is 0.

    Args:
        modes: a tensor containing all 2D modes. Dim 0 and 1 are used for spatial coordinates,
            and dim 2 is the mode index.

    Returns:
        non_orthonormality: A measure for non-orthogonality of the input modes.
        gram: The Gram matrix for the input modes.
    """
    gram = compute_gram(modes.squeeze())
    M = gram.shape[0]
    norm_factor = M
    non_orthonormality = ((gram - torch.eye(*gram.shape)).abs().pow(2)).sum() / norm_factor
    return non_orthonormality, gram


def compute_phase_grad_magsq(amplitude, phase_grad0: tt, phase_grad1: tt, num_of_modes: int) -> tt:
    """
    Compute amplitude-weighted mean of the phase gradient magnitude squared.

    Args:
        amplitude: Tensor containing the amplitude.
        phase_grad0: Phase gradient in dim 0 direction.
        phase_grad1: Phase gradient in dim 1 direction.
        num_of_modes: Number of modes.

    Returns:
        The amplitude-weighted mean of the phase gradient magnitude squared.
    """
    norm_factor = amplitude.sum() * num_of_modes
    phase_grad_magsq = phase_grad0.abs().pow(2) + phase_grad1.abs().pow(2)
    mean_phase_grad_magsq = (amplitude * phase_grad_magsq).sum() / norm_factor
    return mean_phase_grad_magsq


def compute_modes(amplitude: tt, phase_func: callable, phase_kwargs: dict, x: tt, y: tt) -> Tuple[tt, tt, tt]:
    """
    Compute modes

    Compute a set of modes based on given amplitude and phase. The phase is defined by a phase function, x & y
    coordinates, and phase function arguments. The amplitude does not depend on the given x & y coordinates.

    Args:
        amplitude: tensor containing the amplitude.
        phase_func: A function that computes the phase on coordinates x & y and returns a 3D tensor where the last index
            is the mode index.
        phase_kwargs: Keyword arguments for the phase function.
        x: x coordinates for phase function.
        y: y coordinates for phase function.

    Returns:
        modes: Tensor containing the modes (fields). Dim 0 and 1 are for the spatial coordinates. Dim 2 the mode index.
        phase_grad0: Phase gradients in dim 0 direction.
        phase_grad1: Phase gradients in dim 1 direction.
    """
    phase = phase_func(x, y, **phase_kwargs)
    modes = amplitude * torch.exp(1j * phase)

    # Phase grad
    phase_grad0, phase_grad1 = torch.gradient(phase, dim=(0, 1), edge_order=2)
    return modes, phase_grad0, phase_grad1


def plot_mode_optimization(it: int, iterations: int, modes: tt, init_gram: tt, gram: tt, init_non_orthogonality: tt,
                           non_orthogonality: tt, phase_grad_magsq: tt, errors, domain: Dict,
                           non_orthogonalities, phase_grad_magsqs, scale, a, b, poly_powers_x, poly_powers_y,
                           do_plot_all_modes=True, nrows=3, ncols=5,
                           do_save_plot=False, save_path_plot='.', save_filename_plot='mode_optimization_it'):
    # Original Gram matrix
    plt.subplot(nrows-1, 3, 1)
    plt.cla()
    plt.imshow(init_gram.detach().cpu().abs())
    plt.xlabel('mode index')
    plt.ylabel('mode index')
    plt.title(f'Original Gram matrix\nnon-orthogonality = {init_non_orthogonality:.2f}')

    # New Gram matrix
    plt.subplot(nrows-1, 3, 2)
    plt.cla()
    plt.imshow(gram.detach().cpu().abs())
    plt.xlabel('mode index')
    plt.ylabel('mode index')
    plt.title(f'Gram matrix, it {it}\nnon-orthogonality = {non_orthogonality.detach().cpu():.5f}')

    # Error convergence
    plt.subplot(nrows-1, 3, 3)
    plt.cla()
    plt.plot(errors, 'r', label='Error function')
    plt.plot(np.asarray(non_orthogonalities), label='non-orthogonality')
    plt.plot(np.asarray(phase_grad_magsqs), label='mean phase gradient')
    plt.yscale('log')
    plt.xlim((0, iterations))
    plt.xlabel('Iteration')
    plt.legend()
    plt.title('Error convergence')

    if do_plot_all_modes:
        scale = 1 / modes[:, :, 0].abs().max().detach().cpu()

        # Loop over modes
        for i in range(modes.shape[2]):
            plt.subplot(nrows+1, ncols, i + 2*ncols + 1)
            plt.cla()
            plot_field(modes[:, :, i, 0, 0].detach().cpu(), scale=scale,
                       imshow_kwargs={'extent': (domain['x_min'], domain['x_max'], domain['y_min'], domain['y_max'])})
            plt.xticks([])
            plt.yticks([])

            grid_x = 1 + 8 * (domain['x_max'] - domain['x_min'])
            grid_y = 1 + 8 * (domain['y_max'] - domain['y_min'])
            x_grid = torch.linspace(domain['x_min'], domain['x_max'], grid_x).view(1, -1, 1, 1, 1)  # Normalized x
            y_grid = torch.linspace(domain['y_min'], domain['y_max'], grid_y).view(-1, 1, 1, 1, 1)  # Normalized y
            r_mask = x_grid * x_grid + y_grid * y_grid > 1.01
            if a.shape[2] == 1:
                m = 0
            else:
                m = i

            wx_grid, wy_grid = coord_transform(x_grid, y_grid, a[:, :, m:m+1, :, :].detach().cpu(), b[:, :, m:m+1, :, :].detach().cpu(), poly_powers_x, poly_powers_y)
            wx_grid[r_mask] = np.nan
            wy_grid[r_mask] = np.nan
            # Warped arc
            phi_arc = torch.linspace(np.pi / 2, 3 * np.pi / 2, 60)
            x_arc = torch.cos(phi_arc).view(-1, 1, 1, 1, 1)
            y_arc = torch.sin(phi_arc).view(-1, 1, 1, 1, 1)
            wx_arc, wy_arc = coord_transform(x_arc, y_arc, a[:, :, m:m+1, :, :].detach().cpu(), b[:, :, m:m+1, :, :].detach().cpu(), poly_powers_x, poly_powers_y)
            # Plot
            plt.plot(wx_arc.squeeze(), wy_arc.squeeze(), '-', linewidth=0.5)
            plt.plot(wx_grid.squeeze(), wy_grid.squeeze(), '-w', linewidth=0.5)
            plt.plot(wx_grid.squeeze().T, wy_grid.squeeze().T, '-w', linewidth=0.5)
            plt.plot()
            plt.xlim((1.1 * domain['x_min'], 1.1 * domain['x_max']))
            plt.ylim((1.1 * domain['y_min'], 1.1 * domain['y_max']))

    else:       # Plot only a few modes and a transform
        # Example mode 1
        plt.subplot(2, 4, 5)
        plt.cla()
        mode1 = modes[:, :, 0, 0, 0].detach().cpu()
        plot_field(mode1, scale)
        plt.xticks([])
        plt.yticks([])

        # Example mode 2
        plt.subplot(2, 4, 6)
        plt.cla()
        mode2 = modes[:,:,2,0,0].detach().cpu()
        plot_field(mode2, scale)
        plt.xticks([])
        plt.yticks([])

        # Example mode 3
        plt.subplot(2, 4, 7)
        plt.cla()
        mode3 = modes[:,:,3,0,0].detach().cpu()
        plot_field(mode3, scale)
        plt.xticks([])
        plt.yticks([])

        # Show warp function as grid
        plt.subplot(2, 4, 8)
        plt.cla()

        # Warped grid
        ### TODO: make arguments available
        grid_x = 1 + 8 * (domain['x_max'] - domain['x_min'])
        grid_y = 1 + 8 * (domain['y_max'] - domain['y_min'])
        x_grid = torch.linspace(domain['x_min'], domain['x_max'], grid_x).view(1, -1, 1, 1, 1)  # Normalized x coords
        y_grid = torch.linspace(domain['y_min'], domain['y_max'], grid_y).view(-1, 1, 1, 1, 1)  # Normalized y coords
        r_mask = x_grid * x_grid + y_grid * y_grid > 1.01
        wx_grid, wy_grid = coord_transform(x_grid, y_grid, a[:, :, 0:1, :, :].detach().cpu(), b[:, :, 0:1, :, :].detach().cpu(), poly_powers_x, poly_powers_y)
        wx_grid[r_mask] = np.nan
        wy_grid[r_mask] = np.nan
        # Warped arc
        phi_arc = torch.linspace(np.pi / 2, 3 * np.pi / 2, 80)
        x_arc = torch.cos(phi_arc).view(-1, 1, 1, 1, 1)
        y_arc = torch.sin(phi_arc).view(-1, 1, 1, 1, 1)
        wx_arc, wy_arc = coord_transform(x_arc, y_arc, a[:, :, 0:1, :, :].detach().cpu(), b[:, :, 0:1, :, :].detach().cpu(), poly_powers_x, poly_powers_y)
        # Plot
        plt.plot(wx_arc.squeeze(), wy_arc.squeeze(), '-', linewidth=1)
        plt.plot(wx_grid.squeeze(), wy_grid.squeeze(), '-k', linewidth=1)
        plt.plot(wx_grid.squeeze().T, wy_grid.squeeze().T, '-k', linewidth=1)
        plt.plot()
        plt.xlim((1.2 * domain['x_min'], 1.2 * domain['x_max']))
        plt.ylim((1.2 * domain['y_min'], 1.2 * domain['y_max']))
        plt.gca().set_aspect(1)
        plt.xlabel('warped x')
        plt.ylabel('warped y')
        plt.title('Warped pupil coords')

    plt.pause(0.05)

    if do_save_plot:
        plt.savefig(f'{save_path_plot}/{save_filename_plot}{it:04d}.png')


def optimize_modes(domain: dict, amplitude_func: callable, phase_func: callable, amplitude_kwargs: dict = {},
                   phase_kwargs: dict = {}, poly_per_mode: bool = True, poly_powers_x: Tuple = (0, 2, 4, 6),
                   poly_powers_y: Tuple = (0, 2, 4, 6), extra_params: dict = {}, phase_grad_weight: float = 0.1,
                   iterations: int = 500, learning_rate: float = 0.02, optimizer: Optimizer = None, do_plot: bool =
                   True, plot_per_its: int = 10, do_save_plot: bool = False, save_path_plot: str = '.', nrows=3,
                   ncols=5, do_plot_all_modes=True, save_filename_plot='mode_optimization_it'):
    """
    Optimize modes

    Args:
        domain: Dict that specifies x,y limits.
        amplitude_func: Function that returns the amplitude on given x,y coordinates.
        phase_func: Function that returns the phase on given x,y coordinates.
        amplitude_kwargs: Keyword arguments for the amplitude function.
        phase_kwargs: Keyword arguments for the phase function.
        poly_per_mode: If True, each mode will have its own unique transform. If False, one transform is used for every
            mode.
        poly_powers_x: Polynomial powers for x in coordinate transform.
        poly_powers_y: Polynomial powers for y in coordinate transform.
        extra_params: Extra parameters to optimize with the optimization algorithm.
        phase_grad_weight: Weight factor for the phase gradient.
        iterations: Number of iterations for the optimizer.
        learning_rate: Learning rate for the optimizer.
        optimizer: If not None, overrides the default optimizer instance.
        do_plot: Plot intermediate results and convergence graphs.
        plot_per_its: Update plot every this many iterations (if do_plot=True).
        do_save_plot: Save each plot as an image frame (if do_plot=True).
        save_path_plot: Folder path to save each plot.
        nrows: Number of rows in the plot.
        ncols: Number of columns in the plot.
        do_plot_all_modes: If True, plot all modes, instead of a few selected ones.
        save_filename_plot: Filename for each plot. A suffix with the iteration number will be added.

    Returns:
        a: x transform coefficients
        b: y transform coefficients
        new_modes: Array containing the new orthonormal modes
        init_modes: Array containing the initial modes
    """
    # Compute initial coordinates and amplitude profile
    x = torch.linspace(domain['x_min'], domain['x_max'], domain['yxshape'][1]).view(1, -1, 1, 1, 1)  # x coords
    y = torch.linspace(domain['y_min'], domain['y_max'], domain['yxshape'][0]).view(-1, 1, 1, 1, 1)  # y coords
    amplitude_unnorm = amplitude_func(x, y, **amplitude_kwargs)
    amplitude = amplitude_unnorm / amplitude_unnorm.abs().pow(2).sum().sqrt()

    # Compute initial modes
    init_modes_graph, init_phase_grad0_graph, init_phase_grad1_graph = compute_modes(amplitude, phase_func, phase_kwargs, x, y)
    init_modes = init_modes_graph.detach()

    # Determine coefficients shape
    M = init_modes.shape[2]
    Nx = len(poly_powers_x)
    Ny = len(poly_powers_y)
    if poly_per_mode:
        shape = (1, 1, M, Nx, Ny)             # (num_x, num_y, num_modes, poly_degree, poly_degree)
    else:
        shape = (1, 1, 1, Nx, Ny)

    # Initialize coefficient arrays
    a = torch.zeros(shape)
    b = torch.zeros(shape)
    a[0, 0, :, 0, 0] = 1
    b[0, 0, :, 0, 0] = 1
    a.requires_grad = True
    b.requires_grad = True

    # Compute initial values
    init_non_orthogonality, init_gram = compute_non_orthonormality(init_modes)

    # Create dictionary of parameters to optimize
    params = [{'lr': learning_rate, 'params': [a, b]}, {'lr': learning_rate, 'params': extra_params.values()}]

    # Define optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(params, lr=learning_rate, amsgrad=True)
    else:
        optimizer.params = params

    # Initialize arrays for error function values and terms
    errors = [np.nan] * iterations
    non_orthogonalities = [np.nan] * iterations
    phase_grad_magsqs = [np.nan] * iterations
    progress_bar = tqdm(total=iterations)

    # Initialize plot figure
    if do_plot:
        plt.figure(figsize=(16, 9), dpi=90)
        plt.tight_layout()
        plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)

    # Gradient descent loop
    for it in range(iterations):
        # Compute transformed coordinates and modes
        wx, wy = coord_transform(x, y, a, b, poly_powers_x, poly_powers_y)
        new_modes, new_phase_grad0, new_phase_grad1 = compute_modes(amplitude, phase_func, phase_kwargs, wx, wy)

        # Compute error
        non_orthogonality, gram = compute_non_orthonormality(new_modes)
        phase_grad_magsq = compute_phase_grad_magsq(amplitude, new_phase_grad0, new_phase_grad1, M)
        error = non_orthogonality + phase_grad_weight * phase_grad_magsq

        # Save error and terms
        errors[it] = error.detach().cpu()
        non_orthogonalities[it] = non_orthogonality.detach().cpu()
        phase_grad_magsqs[it] = phase_grad_magsq.detach().cpu()

        # Plot
        if do_plot and it % plot_per_its == 0:
            plot_mode_optimization(it=it, iterations=iterations, modes=new_modes, init_gram=init_gram, gram=gram,
                                   init_non_orthogonality=init_non_orthogonality, non_orthogonality=non_orthogonality,
                                   phase_grad_magsq=phase_grad_magsq, errors=errors, domain=domain,
                                   non_orthogonalities=non_orthogonalities, phase_grad_magsqs=phase_grad_magsqs,
                                   scale=60, a=a, b=b, poly_powers_x=poly_powers_x, poly_powers_y=poly_powers_y,
                                   do_save_plot=do_save_plot, save_path_plot=save_path_plot, nrows=nrows, ncols=ncols,
                                   do_plot_all_modes=do_plot_all_modes, save_filename_plot=save_filename_plot)

        # Gradient descent step
        error.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.update()

    if do_plot:
        # Gram matrices and error evolution
        plt.figure(figsize=(13, 4), dpi=120)
        plt.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.13)

        # Original Gram matrix
        plt.subplot(1, 3, 1)
        plt.imshow(init_gram.detach().cpu().abs(), extent=(0.5, M+0.5, M+0.5, 0.5))
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('index n')
        plt.ylabel('index m')
        plt.title(f'a. Initial Gram matrix')
        plt.colorbar()

        # New Gram matrix
        plt.subplot(1, 3, 2)
        plt.imshow(gram.detach().cpu().abs(), extent=(0.5, M+0.5, M+0.5, 0.5))
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('index n')
        plt.ylabel('index m')
        plt.title(f'b. New Gram matrix')
        plt.colorbar()

        # Error convergence
        plt.subplot(1, 3, 3)
        plt.plot(errors, color='tab:red', label='Error=$\\mathcal{N} + w\\mathcal{G}$')
        plt.plot(non_orthogonalities, '--', color='tab:blue', label='$\\mathcal{N}$')
        plt.plot(phase_grad_weight*np.asarray(phase_grad_magsqs), ':', color='tab:green', label='$w\\mathcal{G}$')
        plt.yscale('log')
        plt.xlim((0, iterations))
        plt.xlabel('Iteration')
        plt.legend()
        plt.title('c. Error convergence')

    return a, b, new_modes.squeeze(), init_modes.squeeze()
