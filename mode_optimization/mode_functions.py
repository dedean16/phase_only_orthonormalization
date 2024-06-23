import torch
from torch import Tensor as tt
from torch.optim import Optimizer
import numpy as np
from numpy import ndarray as nd


def inner(a, b, dim):
    return (a * b.conj()).sum(dim=dim)


def field_correlation(a, b, dim):
    aa = inner(a, a, dim)
    bb = inner(b, b, dim)
    ab = inner(a, b, dim)
    return ab / torch.sqrt(aa * bb)


def gaussian(x, y, r_factor):
    r_sq = x ** 2 + y ** 2
    return torch.exp(-(r_factor ** 2 * r_sq)) * (r_sq <= 1)
    # return 1


def warp_func(x, y, a, b, pow_factor = 2):
    """
    x: Nx1x1x1
    y: 1xMx1x1
    a: 1x1xPxQ
    b: 1x1xPxQ
    """
    assert a.shape == b.shape
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)

    # Create arrays of powers
    # Note: a pow_factor 2 means that the indexing is different from the paper by a factor of 2!
    xpows = torch.tensor(range(a.shape[2])).view(1, 1, -1, 1) * pow_factor
    ypows = torch.tensor(range(a.shape[3])).view(1, 1, 1, -1) * pow_factor

    # Raise x and y to even powers
    x_to_powers = x ** xpows
    y_to_powers = y ** ypows

    # Multiply all factors and sum to create the polynomial
    wx = (a * x * x_to_powers * y_to_powers).sum(dim=(2, 3), keepdim=True)
    wy = (b * y * x_to_powers * y_to_powers).sum(dim=(2, 3), keepdim=True)

    return wx, wy


def compute_non_orthogonality(gram):
    return ((gram - torch.eye(*gram.shape)).abs().pow(2)).sum() / gram.shape[0]


def compute_similarity(corrs):
    return corrs.abs().pow(2).sum() / corrs.shape[0]


def compute_gram(modes):
    """
    Compute Gram matrix from collection of modes.

    Args:
        modes: a 3D array containing all 2D modes. Axis 0 and 1 are used for spatial axis,
            and axis 2 is the mode index.
    """
    num_modes = modes.shape[2]
    gram = np.zeros(shape=[num_modes] * 2)

    for n in range(num_modes):
        row_mode = np.expand_dims(modes[:, :, n], 2)
        gram[:, n] = inner(modes, row_mode, dim=(0, 1))

    return gram


def compute_modes(amplitude: tt, phase_func: callable, phase_kwargs: dict, x: tt, y: tt) -> tt:
    """
    Compute modes

    Compute a set of modes based on given amplitude and phase. The phase is defined by a phase function, x & y
    coordinates, and phase function arguments. The amplitude does not depend on the given x & y coordinates.

    Args:
        amplitude: 2D tensor containing the amplitude.
        phase_func: A function that computes the phase on coordinates x & y.
        phase_kwargs: Keyword arguments for the phase function.
        x: x coordinates for phase function.
        y: y coordinates for phase function.
    """
    phase = phase_func(x, y, **phase_args)
    return amplitude * torch.exp(1j * phase)


def optimize_modes(amplitude: tt, phase_func: callable, phase_kwargs: dict = {}, poly_degree: int = 3,
                   poly_per_mode: bool = True, extra_params: dict = {},
                   similarity_weight: float = 0.1, iterations: int = 500, learning_rate: float = 0.02,
                   optimizer: Optimizer = None):
    """
    Optimize modes
    """
    # Compute modes
    init_modes = compute_modes(amplitude, phase_func, phase_args, x, y)

    # Initialize coefficient arrays
    a = torch.zeros((N, N)).view(1, 1, N, N) ### per mode?
    b = torch.zeros((N, N)).view(1, 1, N, N)
    a[0, 0, 0, 0] = 1
    b[0, 0, 0, 0] = 1
    a.requires_grad = True
    b.requires_grad = True

    # Create dictionary of parameters to optimize
    params = {**extra_params}
    params['a'] = a
    params['b'] = b

    # Define optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam([
            {'lr': learning_rate, 'params': params},
        ], lr=learning_rate, amsgrad=True)
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

    for it in range(iterations):
        wx, wy = warp_func(x, y, a, b)
        new_modes = compute_modes(amplitude, phase_func, phase_args, wx, wy)
        gram = compute_gram(new_modes)

        # Compute error
        # non_orthogonality = compute_non_orthogonality ###
        # similarity = compute_similarity ###
        error = non_orthogonality - similarity_weight * similarity

    # Gradient descent step
    error.backward()
    optimizer.step()
    optimizer.zero_grad()

    return a, b, phase_kwargs, new_modes