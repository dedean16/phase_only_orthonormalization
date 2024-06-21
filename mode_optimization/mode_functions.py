import torch
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


def warp_func(x, y, ax, ay):
    """
    x: Nx1x1x1
    y: 1xMx1x1
    a: 1x1xPxQ
    b: 1x1xPxQ
    """
    assert ax.shape == ay.shape
    if isinstance(ax, np.ndarray):
        ax = torch.from_numpy(ax)
    if isinstance(ay, np.ndarray):
        ay = torch.from_numpy(ay)

    # Create arrays of even powers
    # Note that the factor 2 means that the indexing is different from the paper by a factor of two!
    xpows = torch.tensor(range(ax.shape[2])).view(1, 1, -1, 1) * 2
    ypows = torch.tensor(range(ax.shape[3])).view(1, 1, 1, -1) * 2

    # Raise x and y to even powers
    x_to_powers = x**xpows
    y_to_powers = y**ypows

    # Multiply all factors and sum to create the polynomial
    wx = (ax * x * x_to_powers * y_to_powers).sum(dim=(2, 3), keepdim=True)
    wy = (ay * y * x_to_powers * y_to_powers).sum(dim=(2, 3), keepdim=True)

    return wx, wy


def compute_non_orthogonality(overlaps):
    return ((overlaps - torch.eye(*overlaps.shape)).abs().pow(2)).sum() / overlaps.shape[0]


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
    gram = np.zeros(shape=[num_modes]*2)

    for n in range(num_modes):
        row_mode = np.expand_dims(modes[:, :, n], 2)
        gram[:, n] = inner(modes, row_mode, dim=(0, 1))

    return gram
