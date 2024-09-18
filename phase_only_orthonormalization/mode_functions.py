from typing import Tuple, Dict, Sequence

import torch
from torch import Tensor as tt
from torch.optim import Optimizer
import numpy as np
from numpy import ndarray as nd
from tqdm import tqdm
import matplotlib.pyplot as plt

from phase_only_orthonormalization.helper_functions import plot_field, factorial


def inner(A: tt, B: tt, dim: Sequence) -> tt:
    """
    Inner product. This inner product is defined as sum of all element-wise products ∑ᵢ A*ᵢ ⋅ Bᵢ. Ensure to normalize
    A and B correspondingly.

    Args:
        A, B: Tensors containing fields to compute inner product for.
        dim: Tensor dimension to take the inner product over.

    Returns: Inner product (Surprise, surprise!)
    """
    return (A * B.conj()).sum(dim=dim)


def get_coords(domain: dict) -> Tuple[tt, tt]:
    """
    Get x, y coordinates as specified with a domain dictionary.

    Args:
        domain: Dictionary defining the domain and sampling resolution, with keys:
            'x_min': Minimum x value.
            'x_max': Maximum x value.
            'y_min': Minimum y value.
            'y_max': Maximum y value.
            'yxshape': Tuple that defines the number of samples in y and x.

    Returns:
        x, y: Tensors containing coordinates.
    """
    x = torch.linspace(domain['x_min'], domain['x_max'], domain['yxshape'][1]).view(1, -1, 1, 1, 1)  # x coords
    y = torch.linspace(domain['y_min'], domain['y_max'], domain['yxshape'][0]).view(-1, 1, 1, 1, 1)  # y coords
    return x, y


def amplitude_rectangle(x: tt, y: tt, domain: dict):
    """
    Create a normalized constant amplitude function for a rectangular domain.

    Args:
        x: Tensor containing the x spatial coordinates.
        y: Tensor containing the y spatial coordinates.
        domain: Dictionary defining the domain and sampling resolution, with keys:
            'x_min': Minimum x value.
            'x_max': Maximum x value.
            'y_min': Minimum y value.
            'y_max': Maximum y value.
            'yxshape': Tuple that defines the number of samples in y and x.

    Returns: Tensor containing amplitude at the requested coordinates
    """
    value = 1 / np.sqrt(np.prod(domain['yxshape']))
    return value * (x >= domain['x_min']) * (x <= domain['x_max']) * (y >= domain['y_min']) * (y <= domain['y_max'])


def trunc_gaussian(x: tt | nd, y: tt | nd, waist, r_pupil) -> tt:
    """
    Compute a truncated Gaussian on the given x & y coordinates.

    Args:
        x: Tensor containing the x spatial coordinates.
        y: Tensor containing the y spatial coordinates.
        waist: Gaussian waist.
        r_pupil: Pupil radius for truncation.
    
    Returns: Values for truncated Gaussian beam profile.
    """
    if isinstance(x, tt):
        exp = torch.exp
    else:
        exp = np.exp

    r_sq = x ** 2 + y ** 2
    return exp(-(r_sq / waist**2)) * (r_sq <= r_pupil)


def phase_gradient(x, y, kx, ky):
    """
    Phase gradient.

    Args:
        x: Tensor containing the x spatial coordinates.
        y: Tensor containing the y spatial coordinates.
        kx: Wavenumber in x.
        ky: Wavenumber in y.

    Returns:
        A phase gradient of π * (kx*x + ky*y)
    """
    # Ensure Tensors
    if not isinstance(x, tt): x = torch.tensor(x)
    if not isinstance(y, tt): y = torch.tensor(y)
    if not isinstance(kx, tt): kx = torch.tensor(kx)
    if not isinstance(ky, tt): ky = torch.tensor(ky)

    return np.pi * (kx*x + ky*y)


def associated_laguerre_polynomial(x: tt, a, n) -> tt:
    """
    Return the value of the nth associated Laguerre polynomial L_n^a(x).

    Args:
        x: 5D Tensor with input values for the associated Laguerre polynomials L_n^a(x).
        a: Value a.
        n: Value n.

    Returns:
        5D Tensor of values for L_n^a(x), where D is the number of elements of x.
    """
    # Initial associated Laguerre polynomials
    L_list = [None] * max(n+1, 2)

    L_list[0] = torch.ones(x.shape)
    L_list[1] = 1 + a - x

    # Compute the rest of the polynomials using the recurrence relation
    for k in range(1, n):
        L_list[k+1] = ((2*k + 1 + a - x) * L_list[k] - (k + a) * L_list[k-1]) / (k+1)

    return L_list[n]


def laguerre_gauss_mode(x: tt, y: tt, el, p, w0) -> tt:
    """
    Laguerre Gaussian modes.

    Args:
        x: Tensor containing the x spatial coordinates.
        y: Tensor containing the y spatial coordinates.
        el: The topological charge.
        p: The radial order.
        w0: Beam waist.

    Returns:
        Requested Laguerre Gaussian mode.
    """
    z = x + 1j*y
    phi = z.angle()
    r = z.abs()
    RR2 = (r/w0)**2                                                 # Relative Radius Squared
    L = associated_laguerre_polynomial(2*RR2, a=abs(el), n=p)           # associated Laguerre polynomial
    NC = (2*factorial(p) / (np.pi * factorial(p+abs(el)))).sqrt()   # Normalization Constant
    RRPTC = (r*np.sqrt(2) / w0).pow(abs(el))                        # Rel. Radius to the Power of Topological Charge
    return NC * RRPTC * L * torch.exp(-RR2 - 1j*el*phi)


def laguerre_gauss_phase_factor(x: tt, y: tt, el_max, p_max, w0, step_smoothness=0.01, dtype=torch.complex128):
    """
    Phases of Laguerre Gauss modes.

    The phase flips that occur where the Laguerre Gauss amplitude is zero, are smoothed with a smooth step function
    (tanh).

    Args:
        x: Tensor containing the x spatial coordinates.
        y: Tensor containing the y spatial coordinates.
        el_max: The maximum topological charge. This will result in the modes for el = -el_max, -el_max+1, ..., el_max.
        p_max: The maximum radial order. This will result in the modes for p = 0, 1, ..., p_max.
        w0: Beam waist.
        step_smoothness: The amplitude of the Laguerre Gauss modes is normalized. In order to maintain a finite gradient,
            The phase flips are smoothed with a smooth step function: new_abs = tanh(step_steepness * LG.abs()).
        dtype: Data type of the output.

    Returns:
        Phases of requested Laguerre Gaussian mode.
    """
    # Total numbers
    num_el = 2*el_max+1
    num_p = p_max+1
    M = num_p * num_el

    # Initialize phase
    phase_factors = torch.zeros(y.shape[0], x.shape[1], M, 1, 1, dtype=dtype)
    i_mode = 0
    for p in range(p_max+1):                                        # Loop over radial index p
        for el in range(-el_max, el_max + 1):                       # Loop over topological charge
            # Extract coords
            if x.shape[2] > 1:                                      # Each mode has its own transformed coords
                x_mode = x[:, :, i_mode, 0, 0]
                y_mode = y[:, :, i_mode, 0, 0]
            else:                                                   # One transform for all modes
                x_mode = x[:, :, 0, 0, 0]
                y_mode = y[:, :, 0, 0, 0]

            # Mode computation
            LG = laguerre_gauss_mode(x_mode, y_mode, el, p, w0)     # Compute Laguerre Gauss mode

            # Normalize amplitude, but with smooth steps
            phase_factors[:, :, i_mode, 0, 0] = torch.tanh(LG.abs() / step_smoothness) * torch.exp(1j * LG.angle())
            # phase_factors[:, :, i_mode, 0, 0] = torch.sign(LG.abs() / step_smoothness) * torch.exp(1j * LG.angle())
            i_mode += 1

    return phase_factors


def coord_transform(x: tt, y: tt, a: tt | nd, b: tt | nd, p_tuple: Tuple[int, ...], q_tuple: Tuple[int, ...],
                    compute_jacobian: bool = False):
    """
    Coordinate transform.

    Bivariate polynomial coordinate transform of the form:
        x' = x ∑∑ α x^p y^q
        y' = y ∑∑ ϐ x^p y^q
    where α,ϐ contain the polynomial coefficients, and have indices m, p, q. m is the mode index. The double sum ∑∑ run
    over all p, q.

    Dim 0 and 1 are used for spatial coordinate index. Dim 2 for mode index. Dim 3 and 4 for polynomial power index.

    Args:
        x: 5D Tensor containing the x spatial coordinates.
        y: 5D Tensor containing the y spatial coordinates.
        a: 5D Tensor containing the x polynomial coefficients (alpha)
        b: 5D Tensor containing the y polynomial coefficients (beta)
        p_tuple: list of polynomial powers for x
        q_tuple: list of polynomial powers for y
        compute_jacobian: If True, also compute and return the

    Returns:
        wx, wy: The transformed (warped) coordinates x' and y'.
        jacobian: Returned if compute_jacobian is True. Jacobian of the transformation.
    """
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)
    assert a.shape == b.shape

    # Create arrays of powers
    p = torch.tensor(p_tuple).view(1, 1, 1, -1, 1)    # p
    q = torch.tensor(q_tuple).view(1, 1, 1, 1, -1)    # q

    # Raise x and y to even powers
    x_pow_p = x ** p    # x^p
    y_pow_q = y ** q    # y^q

    # Multiply all factors and sum to create the polynomial
    wx = (a * x * x_pow_p * y_pow_q).sum(dim=(3, 4), keepdim=True)
    wy = (b * y * x_pow_p * y_pow_q).sum(dim=(3, 4), keepdim=True)

    # Compute and return Jacobian, only if requested
    if compute_jacobian:
        # Derivatives
        dwx_dx = ((p+1) * a * x_pow_p * y_pow_q).sum(dim=(3, 4), keepdim=True)
        dwx_dy = (q * x * a * x_pow_p * y**(q-1)).sum(dim=(3, 4), keepdim=True)
        dwy_dy = ((q+1) * b * y_pow_q * x_pow_p).sum(dim=(3, 4), keepdim=True)
        dwy_dx = (p * y * b * y_pow_q * x**(p-1)).sum(dim=(3, 4), keepdim=True)

        jacobian = dwx_dx * dwy_dy - dwx_dy * dwy_dx
        return wx, wy, jacobian

    else:
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
    phase_grad_magsq = phase_grad0.abs().pow(2) + phase_grad1.abs().pow(2)
    mean_phase_grad_magsq = (amplitude.abs().pow(2) * phase_grad_magsq).sum() / num_of_modes
    return mean_phase_grad_magsq


def compute_modes(amplitude: tt, phase_func: callable, phase_kwargs: dict, x: tt, y: tt,
                  phase_factor_func: callable = None) -> Tuple[tt, tt, tt]:
    """
    Compute modes

    Compute a set of modes based on given amplitude and phase. The phase is defined by a phase function, x & y
    coordinates, and phase function arguments. The amplitude does not depend on the given x & y coordinates.

    Args:
        amplitude: tensor containing the amplitude.
        phase_func: A function that computes the phase on coordinates x & y and returns a 3D tensor where the last index
            is the mode index. When None, phase_factor_func is used to determine the phase instead.
        phase_kwargs: Keyword arguments for the phase function.
        x: x coordinates for phase function.
        y: y coordinates for phase function.
        phase_factor_func: A function that computes the phase factor exp(i𝜙) on coordinates x & y and returns a 3D
            tensor where the last index is the mode index. Only used when phase_func is None.
            instead.

    Returns:
        modes: Tensor containing the modes (fields). Dim 0 and 1 are for the spatial coordinates. Dim 2 the mode index.
        phase_grad0: Phase gradients in dim 0 direction.
        phase_grad1: Phase gradients in dim 1 direction.
    """
    if phase_func is None:
        phase_factor = phase_factor_func(x, y, **phase_kwargs)
        phase = phase_factor.angle()
    else:
        phase = phase_func(x, y, **phase_kwargs)
        phase_factor = torch.exp(1j * phase)
    modes = amplitude * phase_factor

    # Phase grad
    phase_grad0, phase_grad1 = torch.gradient(phase, dim=(0, 1), edge_order=2)
    return modes, phase_grad0, phase_grad1


def plot_mode_optimization(it: int, iterations: int, modes: tt, init_gram: tt, gram: tt, init_non_orthogonality: tt,
                           non_orthogonality: tt, phase_grad_magsq: tt, errors, domain: Dict,
                           non_orthogonalities, phase_grad_magsqs, scale, a, b, p_tuple, q_tuple,
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

        # Loop over modes
        for i in range(modes.shape[2]):
            plt.subplot(nrows+1, ncols, i + 2*ncols + 1)
            plt.cla()
            scale = 1 / modes[:, :, i, 0, 0].abs().max().detach().cpu()
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

            wx_grid, wy_grid = coord_transform(x_grid, y_grid, a[:, :, m:m+1, :, :].detach().cpu(), b[:, :, m:m+1, :, :].detach().cpu(), p_tuple, q_tuple)
            wx_grid[r_mask] = np.nan
            wy_grid[r_mask] = np.nan
            # Warped arc
            phi_arc = torch.linspace(np.pi / 2, 3 * np.pi / 2, 60)
            x_arc = torch.cos(phi_arc).view(-1, 1, 1, 1, 1)
            y_arc = torch.sin(phi_arc).view(-1, 1, 1, 1, 1)
            wx_arc, wy_arc = coord_transform(x_arc, y_arc, a[:, :, m:m+1, :, :].detach().cpu(), b[:, :, m:m+1, :, :].detach().cpu(), p_tuple, q_tuple)
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
        wx_grid, wy_grid = coord_transform(x_grid, y_grid, a[:, :, 0:1, :, :].detach().cpu(), b[:, :, 0:1, :, :].detach().cpu(), p_tuple, q_tuple)
        wx_grid[r_mask] = np.nan
        wy_grid[r_mask] = np.nan
        # Warped arc
        phi_arc = torch.linspace(np.pi / 2, 3 * np.pi / 2, 80)
        x_arc = torch.cos(phi_arc).view(-1, 1, 1, 1, 1)
        y_arc = torch.sin(phi_arc).view(-1, 1, 1, 1, 1)
        wx_arc, wy_arc = coord_transform(x_arc, y_arc, a[:, :, 0:1, :, :].detach().cpu(), b[:, :, 0:1, :, :].detach().cpu(), p_tuple, q_tuple)
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


def optimize_modes(domain: dict, amplitude_func: callable, phase_func: callable,
                   amplitude_kwargs: dict = {}, phase_kwargs: dict = {}, phase_factor_func: callable = None,
                   poly_per_mode: bool = True, p_tuple: Tuple = (0, 2, 4, 6),
                   q_tuple: Tuple = (0, 2, 4, 6), extra_params: dict = {}, phase_grad_weight: float = 0.1,
                   iterations: int = 500, learning_rate: float = 0.02, optimizer: Optimizer = None, do_plot: bool =
                   True, plot_per_its: int = 10, do_save_plot: bool = False, save_path_plot: str = '.', nrows=3,
                   ncols=5, do_plot_all_modes=True, save_filename_plot='mode_optimization_it'):
    """
    Optimize modes.

    Orthonormalize a set of 2D functions where every function has the same amplitude profile.

    Note: The returned fields are normalized for the inner product ∑ᵢ A*ᵢ Bᵢ. To get the functions normalized for the
    area integral inner product ∬ A*(x,y) B(x,y) dx dy, multiply the fields by a factor of num_samples / domain area.

    Args:
        domain: Dict that specifies x,y limits and sampling, see get_coords documentation for detailed info.
        amplitude_func: Function that returns the amplitude on given x,y coordinates.
        phase_func: A function that computes the phase on coordinates x & y for all modes. When None, phase_factor_func
            is used to determine the phase instead.
        amplitude_kwargs: Keyword arguments for the amplitude function.
        phase_kwargs: Keyword arguments for the phase function.
        phase_factor_func: A function that computes the phase factor exp(i𝜙) on coordinates x & y and returns a
            tensor where the last index is the mode index. Only used when phase_func is None.
        poly_per_mode: If True, each mode will have its own unique transform. If False, one transform is used for every
            mode.
        p_tuple: Polynomial powers for x for coordinate transform.
        q_tuple: Polynomial powers for y for coordinate transform.
        extra_params: Extra parameters to optimize with the optimization algorithm.
        phase_grad_weight: Weight factor for the phase gradient. 1/w² in the paper.
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
    x, y = get_coords(domain)
    amplitude_unnorm = amplitude_func(x, y, **amplitude_kwargs)
    amplitude = amplitude_unnorm / amplitude_unnorm.abs().pow(2).sum().sqrt()

    # Compute initial modes
    init_modes_graph, init_phase_grad0_graph, init_phase_grad1_graph = \
        compute_modes(amplitude, phase_func, phase_kwargs, x, y, phase_factor_func=phase_factor_func)
    init_modes = init_modes_graph.detach()

    # Determine coefficients shape
    M = init_modes.shape[2]
    Nx = len(p_tuple)
    Ny = len(q_tuple)
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
        wx, wy = coord_transform(x, y, a, b, p_tuple, q_tuple)
        new_modes, new_phase_grad0, new_phase_grad1 = \
            compute_modes(amplitude, phase_func, phase_kwargs, wx, wy, phase_factor_func=phase_factor_func)

        # Compute error
        non_orthogonality, gram = compute_non_orthonormality(new_modes)
        phase_grad_magsq = compute_phase_grad_magsq(amplitude, new_phase_grad0, new_phase_grad1, M)
        error = non_orthogonality + phase_grad_weight * phase_grad_magsq

        ###
        print()
        print(gram.abs().min().detach().item())
        print(gram.abs().max().detach().item())
        ###

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
                                   scale=60, a=a, b=b, p_tuple=p_tuple, q_tuple=q_tuple,
                                   do_save_plot=do_save_plot, save_path_plot=save_path_plot, nrows=nrows, ncols=ncols,
                                   do_plot_all_modes=do_plot_all_modes, save_filename_plot=save_filename_plot)

        # Gradient descent step
        error.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.update()

    if do_plot:
        # Gram matrices and error evolution
        plt.figure(figsize=(14, 4), dpi=120)
        plt.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.13)

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
        plt.plot(errors, color='tab:red', label='Error=$\\mathcal{N} + \\mathcal{G}/w^2$')
        plt.plot(non_orthogonalities, '--', color='tab:blue', label='$\\mathcal{N}$')
        plt.plot(phase_grad_weight*np.asarray(phase_grad_magsqs), ':', color='tab:green', label='$\\mathcal{G}/w^2$')
        plt.yscale('log')
        plt.xlim((0, iterations))
        plt.xlabel('Iteration')
        plt.legend()
        plt.title('c. Error convergence')

        plt.show()

    return a, b, new_modes.squeeze(), init_modes.squeeze()
