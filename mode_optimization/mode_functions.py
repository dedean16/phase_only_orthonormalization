from typing import Tuple

import torch
from torch import Tensor as tt
from torch.optim import Optimizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from helper_functions import plot_field, mse


def inner(a, b, dim):
    return (a * b.conj()).sum(dim=dim)


def apo_gaussian(x: tt, y: tt, waist, r_pupil):
    """
    Compute an apodized gaussian on the given x & y coordinates.

    Args:
        x
        y
        waist
        r_pupil
    """
    r_sq = x ** 2 + y ** 2
    return torch.exp(-(r_sq / waist**2)) * (r_sq <= r_pupil)


def warp_func(x, y, a, b, pow_factor):
    """
    x: Xx1x1x1x1
    y: 1xYx1x1x1
    a: 1x1xMxNxN or 1x1x1xNxN
    b: 1x1xMxNxN or 1x1x1xNxN
    """
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)
    assert a.shape == b.shape

    # Create arrays of powers
    # Note: a pow_factor 2 means that the indexing is different from the paper by a factor of 2!
    xpows = torch.tensor(range(a.shape[3])).view(1, 1, 1, -1, 1) * pow_factor
    ypows = torch.tensor(range(a.shape[4])).view(1, 1, 1, 1, -1) * pow_factor

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
        modes: a 3D tensor containing all 2D modes. Axis 0 and 1 are used for spatial axis,
            and axis 2 is the mode index.

    Returns:
        gram: Gram matrix
    """
    num_modes = modes.shape[2]
    gram = torch.zeros(num_modes, num_modes, dtype=modes.dtype)

    for n in range(num_modes):
        row_mode = modes[:, :, n:n+1]
        gram[:, n] = inner(modes, row_mode, dim=(0, 1))

    return gram


def compute_non_orthogonality(modes: tt) -> Tuple[tt, tt]:
    """
    Compute non-orthogonality

    Compute the non-orthogonality of a set of normalized modes. For orthonormal bases, this value is 0.

    Args:
        modes: a 3D tensor containing all 2D modes. Axis 0 and 1 are used for spatial axis,
            and axis 2 is the mode index.

    Returns:
        non_orthogonality: A measure for non-orthogonality of the input modes.
        gram: The Gram matrix for the input modes.
    """
    gram = compute_gram(modes.squeeze())
    M = gram.shape[0]
    # norm_factor = M * (M-1)
    norm_factor = M
    non_orthogonality = ((gram - torch.eye(*gram.shape)).abs().pow(2)).sum() / norm_factor
    return non_orthogonality, gram


def compute_similarity(modes1: tt, modes2: tt) -> tt:
    """
    Compute the similarity between two sets of normalized modes. If modes1 and modes2 are equal and normalized,
    this function returns 1.

    Args:
        modes1 and modes2: 3D arrays containing all 2D modes. Axis 0 and 1 are used for spatial axis,
            and axis 2 is the mode index.

    Returns:
        similarity
    """
    return inner(modes1, modes2, dim=(0, 1)).abs().sum() / modes1.shape[2]


def compute_phase_grad_mse(amplitude, init_phase_grad0: tt, init_phase_grad1: tt, new_phase_grad0: tt, new_phase_grad1: tt) -> tt:
    """
    Compute mode-mean squared error of x,y-mean squared phase gradients

    Args:

    Returns:

    """
    amp_sum = amplitude.sum(dim=(0, 1))
    init_phase_grad_square = init_phase_grad0.abs().pow(2) + init_phase_grad1.abs().pow(2)
    init_mean_phase_grad = (amplitude * init_phase_grad_square).sum(dim=(0, 1)) / amp_sum
    new_phase_grad_square = new_phase_grad0.abs().pow(2) + new_phase_grad1.abs().pow(2)
    new_mean_phase_grad = (amplitude * new_phase_grad_square.sum(dim=(0, 1))) / amp_sum
    # mse = (init_mean_phase_grad - new_mean_phase_grad).abs().pow(2).mean()
    me = (new_mean_phase_grad / (init_mean_phase_grad + 1e-6) - 1).abs().mean()
    return me


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
        modes:
        phase_grad0:
        phase_grad1:
    """
    phase = phase_func(x, y, **phase_kwargs)
    modes = amplitude * torch.exp(1j * phase)

    # Phase grad
    phase_grad0, phase_grad1 = torch.gradient(phase, dim=(0, 1), edge_order=2)
    return modes, phase_grad0, phase_grad1


def plot_mode_optimization(it: int, iterations: int, modes: tt, init_gram: tt, gram: tt, init_non_orthogonality: tt,
                           non_orthogonality: tt, init_similarity: tt, similarity: tt, phase_grad_mse: tt, errors,
                           non_orthogonalities, similarities, phase_grad_mses, scale, a, b, pow_factor,
                           do_plot_all_modes=True, nrows=3, ncols=5,
                           do_save_plot=False, save_path_plot='.', save_filename_plot='mode_optimization_it'):
    # Original Gram matrix
    plt.subplot(nrows-1, 4, 1)
    plt.cla()
    plt.imshow(init_gram.detach().cpu().abs())
    plt.xlabel('mode index')
    plt.ylabel('mode index')
    plt.title(f'Original Gram matrix\nnon-orthogonality = {init_non_orthogonality:.2f}')

    # New Gram matrix
    plt.subplot(nrows-1, 4, 2)
    plt.cla()
    plt.imshow(gram.detach().cpu().abs())
    plt.xlabel('mode index')
    plt.ylabel('mode index')
    plt.title(f'Gram matrix, it {it}\nnon-orthogonality = {non_orthogonality.detach().cpu():.5f}')

    # Error convergence
    plt.subplot(nrows-1, 4, 3)
    plt.cla()
    plt.plot(errors, 'r', label='Error function')
    plt.xlim((0, iterations))
    plt.xlabel('Iteration')
    plt.ylim((min([*errors, 0]), max(errors)))
    plt.legend()
    plt.title('Error convergence')

    # Error term evolution
    plt.subplot(nrows-1, 4, 4)
    plt.cla()
    plt.plot(np.asarray(non_orthogonalities), label='non-orthogonality')
    plt.plot(similarities, label='similarity')
    plt.plot(np.asarray(phase_grad_mses)*10, label='mean phase gradient x10')
    plt.xlim((0, iterations))
    plt.xlabel('Iteration')
    plt.ylim((0, 1))
    plt.legend()
    plt.title('Error terms')

    if do_plot_all_modes:
        scale = 1 / modes[:, :, 0].abs().max().detach().cpu()

        # Loop over modes
        for i in range(modes.shape[2]):
            plt.subplot(nrows+1, ncols, i + 2*ncols + 1)
            plt.cla()
            plot_field(modes[:, :, i, 0, 0].detach().cpu(), scale=scale)
            plt.xticks([])
            plt.yticks([])

    else:       # Plot only a few modes and a transform
        # Example mode 1
        plt.subplot(2, 4, 5)
        plt.cla()
        mode1 = modes[:,:,0,0,0].detach().cpu()
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
        x_grid = torch.linspace(-1, 0, 11).view(1, -1, 1, 1, 1)  # Normalized x coords
        y_grid = torch.linspace(-1, 1, 21).view(-1, 1, 1, 1, 1)  # Normalized y coords
        r_mask = x_grid * x_grid + y_grid * y_grid > 1.01
        wx_grid, wy_grid = warp_func(x_grid, y_grid, a[:, :, 0:1, :, :].detach().cpu(), b[:, :, 0:1, :, :].detach().cpu(), pow_factor=pow_factor)
        wx_grid[r_mask] = np.nan
        wy_grid[r_mask] = np.nan
        # Warped arc
        phi_arc = torch.linspace(np.pi / 2, 3 * np.pi / 2, 80)
        x_arc = torch.cos(phi_arc).view(-1, 1, 1, 1, 1)
        y_arc = torch.sin(phi_arc).view(-1, 1, 1, 1, 1)
        wx_arc, wy_arc = warp_func(x_arc, y_arc, a[:, :, 0:1, :, :].detach().cpu(), b[:, :, 0:1, :, :].detach().cpu(), pow_factor=pow_factor)
        # Plot
        plt.plot(wx_arc.squeeze(), wy_arc.squeeze(), '-', linewidth=1)
        plt.plot(wx_grid.squeeze(), wy_grid.squeeze(), '-k', linewidth=1)
        plt.plot(wx_grid.squeeze().T, wy_grid.squeeze().T, '-k', linewidth=1)
        plt.plot()
        plt.xlim((-1.25, 0.1))
        plt.ylim((-1.25, 1.25))
        plt.gca().set_aspect(1)
        plt.xlabel('warped x')
        plt.ylabel('warped y')
        plt.title('Warped pupil coords')

    plt.pause(0.02)

    if do_save_plot:
        plt.savefig(f'{save_path_plot}/{save_filename_plot}{it:04d}.png')


def optimize_modes(domain: dict, amplitude_func: callable, phase_func: callable, amplitude_kwargs: dict = {},
                   phase_kwargs: dict = {}, poly_degree: int = 3, poly_per_mode: bool = True, pow_factor=2,
                   extra_params: dict = {}, similarity_weight: float = 0.1, phase_grad_weight: float = 0.1,
                   iterations: int = 500, learning_rate: float = 0.02, optimizer: Optimizer = None,
                   do_plot: bool = True, plot_per_its: int = 10, do_save_plot: bool = False, save_path_plot: str = '.',
                   nrows=3, ncols=5, do_plot_all_modes=True, save_filename_plot='mode_optimization_it'):
    """
    Optimize modes
    """
    # Compute initial coordinates and amplitude profile
    x = torch.linspace(domain['x_min'], domain['x_max'], domain['yxshape'][1]).view(1, -1, 1, 1, 1)  # x coords
    y = torch.linspace(domain['y_min'], domain['y_max'], domain['yxshape'][0]).view(-1, 1, 1, 1, 1)  # y coords
    amplitude_unnorm = apo_gaussian(x, y, **amplitude_kwargs)
    amplitude = amplitude_unnorm / amplitude_unnorm.abs().pow(2).sum().sqrt()

    # Compute initial modes
    init_modes_graph, init_phase_grad0_graph, init_phase_grad1_graph = compute_modes(amplitude, phase_func, phase_kwargs, x, y)
    init_modes = init_modes_graph.detach()
    init_phase_grad0 = init_phase_grad0_graph.detach()
    init_phase_grad1 = init_phase_grad1_graph.detach()

    # Determine coefficients shape
    M = init_modes.shape[2]
    N = poly_degree
    if poly_per_mode:
        shape = (1, 1, M, N, N)             # (num_x, num_y, num_modes, poly_degree, poly_degree)
    else:
        shape = (1, 1, 1, N, N)

    # Initialize coefficient arrays
    a = torch.zeros(shape)
    b = torch.zeros(shape)
    a[0, 0, :, 0, 0] = 1
    b[0, 0, :, 0, 0] = 1
    a.requires_grad = True
    b.requires_grad = True

    # Compute initial values
    init_non_orthogonality, init_gram = compute_non_orthogonality(init_modes)
    init_similarity = compute_similarity(init_modes, init_modes)

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
    similarities = [np.nan] * iterations
    phase_grad_mses = [np.nan] * iterations
    progress_bar = tqdm(total=iterations)

    # Initialize plot figure
    if do_plot:
        plt.figure(figsize=(16, 9), dpi=90)
        plt.tight_layout()
        plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)

    # Gradient descent loop
    for it in range(iterations):
        wx, wy = warp_func(x, y, a, b, pow_factor=pow_factor)
        new_modes, new_phase_grad0, new_phase_grad1 = compute_modes(amplitude, phase_func, phase_kwargs, wx, wy)

        # Compute error
        non_orthogonality, gram = compute_non_orthogonality(new_modes)
        similarity = compute_similarity(new_modes, init_modes)
        phase_grad_mse = compute_phase_grad_mse(amplitude, init_phase_grad0, init_phase_grad1, new_phase_grad0, new_phase_grad1)
        error = non_orthogonality - similarity_weight * similarity + phase_grad_weight * phase_grad_mse

        # Save error and terms
        errors[it] = error.detach().cpu()
        non_orthogonalities[it] = non_orthogonality.detach().cpu()
        similarities[it] = similarity.detach().cpu()
        phase_grad_mses[it] = phase_grad_mse.detach().cpu()

        if do_plot and it % plot_per_its == 0:
            plot_mode_optimization(it=it, iterations=iterations, modes=new_modes, init_gram=init_gram, gram=gram,
                                   init_non_orthogonality=init_non_orthogonality, non_orthogonality=non_orthogonality,
                                   init_similarity=init_similarity, similarity=similarity,
                                   phase_grad_mse=phase_grad_mse, errors=errors,
                                   non_orthogonalities=non_orthogonalities, similarities=similarities,
                                   phase_grad_mses=phase_grad_mses, scale=50, a=a, b=b, pow_factor=pow_factor,
                                   do_save_plot=do_save_plot, save_path_plot=save_path_plot, nrows=nrows, ncols=ncols,
                                   do_plot_all_modes=do_plot_all_modes, save_filename_plot=save_filename_plot)

        # Gradient descent step
        error.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.update()

    return a, b, new_modes.squeeze(), init_modes.squeeze()
