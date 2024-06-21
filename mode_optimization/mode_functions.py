import torch
import numpy as np

from openwfs.algorithms.basic_fourier import build_square_k_space


def inner(a, b, dim):
    return (a * b.conj()).sum(dim=dim)


def field_correlation(a, b, dim):
    aa = inner(a, a, dim)
    bb = inner(b, b, dim)
    ab = inner(a, b, dim)
    return ab / torch.sqrt(aa * bb)


def tilt(x, y, kx, ky):
    return torch.exp(1j * np.pi * (kx*x + ky*y))


def beam_profile(x, y, r_factor):
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


def compute_mode(mode_shape, k, r_factor, ax=None, ay=None, x_min=-1, x_max=0, y_min=-1, y_max=1, ignore_warp=False,
                 ignore_amplitude=False):
    # Coordinates
    x = torch.linspace(x_min, x_max, mode_shape[1] // 2).view(1, -1, 1, 1)  # Normalized x coords
    y = torch.linspace(y_min, y_max, mode_shape[0]).view(-1, 1, 1, 1)  # Normalized y coords

    # Amplitude
    if ignore_amplitude:
        amplitude = 1
    else:
        amplitude = beam_profile(x, y, r_factor)

    # Phase
    if ignore_warp:
        phase_factor = tilt(x, y, k[1], k[0])
    else:
        wx, wy = warp_func(x, y, ax, ay)
        phase_factor = tilt(wx, wy, k[1], k[0])
    return amplitude * phase_factor


def compute_gram(r_factor, ax, ay, shape, k_min, k_max):
    # Initializations
    k_space = build_square_k_space(k_min=k_min, k_max=k_max)
    overlaps = torch.zeros(([k_space.shape[1]] * 2), dtype=torch.complex128)
    tilt_corrs = torch.zeros([k_space.shape[1]], dtype=torch.complex128)

    # Compute overlaps
    for i1, k1 in enumerate(k_space.T):
        warped_mode1 = compute_mode(shape, k1, r_factor, ax, ay)

        # plot_field(warped_mode1.squeeze().detach(), 1)
        # plt.title('Mode')
        # complex_colorbar(1)
        # plt.pause(0.1)

        for i2, k2 in enumerate(k_space.T):
            warped_mode2 = compute_mode(shape, k2, r_factor, ax, ay)
            overlaps[i1, i2] = field_correlation(warped_mode1, warped_mode2, dim=(0, 1, 2, 3))

        tilt_mode1 = compute_mode(shape, k1, r_factor, ignore_warp=True)
        tilt_corrs[i1] = field_correlation(warped_mode1, tilt_mode1, dim=(0, 1, 2, 3))

    return overlaps, tilt_corrs


def compute_non_orthogonality(overlaps):
    return ((overlaps - torch.eye(*overlaps.shape)).abs().pow(2)).sum() / overlaps.shape[0]


def compute_tilt_similarity(tilt_corrs):
    return tilt_corrs.abs().pow(2).sum() / tilt_corrs.shape[0]
