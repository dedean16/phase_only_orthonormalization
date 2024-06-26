from typing import Tuple

import torch
from torch import Tensor as tt
from torch.optim import Optimizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from helper_functions import plot_field


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
    return inner(modes1, modes2, dim=(0, 1)).abs().pow(2).sum() / modes1.shape[2]


def compute_modes(amplitude: tt, phase_func: callable, phase_kwargs: dict, x: tt, y: tt) -> tt:
    """
    Compute modes

    Compute a set of modes based on given amplitude and phase. The phase is defined by a phase function, x & y
    coordinates, and phase function arguments. The amplitude does not depend on the given x & y coordinates.

    Args:
        amplitude: 2D tensor containing the amplitude.
        phase_func: A function that computes the phase on coordinates x & y and returns a 3D tensor where the last index
            is the mode index.
        phase_kwargs: Keyword arguments for the phase function.
        x: x coordinates for phase function.
        y: y coordinates for phase function.

    Returns:

    """
    phase = phase_func(x, y, **phase_kwargs)
    modes = amplitude * torch.exp(1j * phase)
    return modes


def plot_mode_optimization(it: int, iterations: int, modes: tt, init_gram: tt, gram: tt, init_non_orthogonality: tt,
                           non_orthogonality: tt, init_similarity: tt, similarity: tt, errors, non_orthogonalities,
                           similarities, scale, a, b, pow_factor, do_save_plot=False, save_path_plot='.'):
    # Original Gram matrix
    plt.subplot(2, 4, 1)
    plt.cla()
    plt.imshow(init_gram.detach().abs())
    plt.xlabel('$k_1$ linear index')
    plt.ylabel('$k_2$ linear index')
    plt.title(f'Original Gram matrix (normalized)\nnon-orthogonality = {init_non_orthogonality*100:.3f}%')

    # New Gram matrix
    plt.subplot(2, 4, 2)
    plt.cla()
    plt.imshow(gram.detach().abs())
    plt.xlabel('$k_1$ linear index')
    plt.ylabel('$k_2$ linear index')
    plt.title(f'Gram matrix (normalized), it {it}\nnon-orthogonality = {non_orthogonality.detach()*100:.3f}%')

    # Error convergence
    plt.subplot(2, 4, 3)
    plt.cla()
    plt.plot(errors, 'r', label='Error function')
    plt.xlim((0, iterations))
    plt.xlabel('Iteration')
    plt.ylim((min([*errors, 0]), max(errors)))
    plt.legend()
    plt.title('Error convergence')

    # Error term evolution
    plt.subplot(2, 4, 4)
    plt.cla()
    plt.plot(np.asarray(non_orthogonalities)*100, label='non-orthogonality (%)')
    plt.plot(similarities, label='similarity')
    plt.xlim((0, iterations))
    plt.xlabel('Iteration')
    plt.ylim((0, 1))
    plt.legend()
    plt.title('Error terms')

    # Example mode 1
    plt.subplot(2, 4, 5)
    plt.cla()
    mode1 = modes[:,:,0,0,0].detach()
    plot_field(mode1, scale)
    plt.xticks([])
    plt.yticks([])

    # Example mode 2
    plt.subplot(2, 4, 6)
    plt.cla()
    mode2 = modes[:,:,1,0,0].detach()
    plot_field(mode2, scale)
    plt.xticks([])
    plt.yticks([])

    # Example mode 3
    plt.subplot(2, 4, 7)
    plt.cla()
    mode3 = modes[:,:,2,0,0].detach()
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
    wx_grid, wy_grid = warp_func(x_grid, y_grid, a[:, :, 0:1, :, :].detach(), b[:, :, 0:1, :, :].detach(), pow_factor=pow_factor)
    wx_grid[r_mask] = np.nan
    wy_grid[r_mask] = np.nan
    # Warped arc
    phi_arc = torch.linspace(np.pi / 2, 3 * np.pi / 2, 80)
    x_arc = torch.cos(phi_arc).view(-1, 1, 1, 1, 1)
    y_arc = torch.sin(phi_arc).view(-1, 1, 1, 1, 1)
    wx_arc, wy_arc = warp_func(x_arc, y_arc, a[:, :, 0:1, :, :].detach(), b[:, :, 0:1, :, :].detach(), pow_factor=pow_factor)
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
    plt.pause(1e-2)

    if do_save_plot:
        plt.savefig(f'{save_path_plot}/mode_optimization_it{it:04d}.png')


def optimize_modes(domain: dict, amplitude_func: callable, phase_func: callable, amplitude_kwargs: dict = {},
                   phase_kwargs: dict = {}, poly_degree: int = 3, poly_per_mode: bool = True, pow_factor = 2,
                   extra_params: dict = {}, similarity_weight: float = 0.1, iterations: int = 500,
                   learning_rate: float = 0.02, optimizer: Optimizer = None, do_plot: bool = True,
                   plot_per_its: int = 10, do_save_plot: bool = False, save_path_plot: str = '.'):
    """
    Optimize modes
    """
    # Compute initial coordinates and amplitude profile
    x = torch.linspace(domain['x_min'], domain['x_max'], domain['yxshape'][1]).view(1, -1, 1, 1, 1)  # x coords
    y = torch.linspace(domain['y_min'], domain['y_max'], domain['yxshape'][0]).view(-1, 1, 1, 1, 1)  # y coords
    amplitude_unnorm = apo_gaussian(x, y, **amplitude_kwargs)
    amplitude = amplitude_unnorm / amplitude_unnorm.abs().pow(2).sum().sqrt()

    # Compute initial modes
    init_modes = compute_modes(amplitude, phase_func, phase_kwargs, x, y)

    # ####
    # import matplotlib.pyplot as plt
    # from helper_functions import plot_field
    # scale = 50
    # plt.figure()
    # plot_field(init_modes[:, :, 1, 0, 0].detach(), scale=scale)
    # plt.show()
    # ####

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
    params = [{'lr': learning_rate, 'params': [a, b]}, {'lr': learning_rate, 'params': extra_params}]

    # Define optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(params, lr=learning_rate, amsgrad=True)
    else:
        optimizer.params = params

    # Initialize arrays for error function values and terms
    errors = [np.nan] * iterations
    non_orthogonalities = [np.nan] * iterations
    similarities = [np.nan] * iterations
    progress_bar = tqdm(total=iterations)

    # Initialize plot figure
    if do_plot:
        plt.figure(figsize=(16, 10), dpi=90)
        plt.tight_layout()

    # Gradient descent loop
    for it in range(iterations):
        wx, wy = warp_func(x, y, a, b, pow_factor=pow_factor)
        new_modes = compute_modes(amplitude, phase_func, phase_kwargs, wx, wy)

        # Compute error
        non_orthogonality, gram = compute_non_orthogonality(new_modes)
        similarity = compute_similarity(new_modes, init_modes)
        error = non_orthogonality - similarity_weight * similarity

        # Save error and terms
        errors[it] = error.detach()
        non_orthogonalities[it] = non_orthogonality.detach()
        similarities[it] = similarity.detach()

        if do_plot and it % plot_per_its == 0:
           plot_mode_optimization(it=it, iterations=iterations, modes=new_modes, init_gram=init_gram, gram=gram,
                                  init_non_orthogonality=init_non_orthogonality, non_orthogonality=non_orthogonality,
                                  init_similarity=init_similarity, similarity=similarity, errors=errors,
                                  non_orthogonalities=non_orthogonalities, similarities=similarities, scale=50, a=a,
                                  b=b, pow_factor=pow_factor, do_save_plot=do_save_plot, save_path_plot=save_path_plot)

        # Gradient descent step
        error.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.update()

    return a, b, new_modes.squeeze(), init_modes.squeeze()
