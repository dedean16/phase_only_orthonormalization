import torch
import numpy as np

from mode_functions import inner


def field_correlation(a, b, dim):
    aa = inner(a, a, dim)
    bb = inner(b, b, dim)
    ab = inner(a, b, dim)
    return ab / torch.sqrt(aa * bb)


def tilt(x, y, kx, ky):
    return torch.exp(1j * np.pi * (kx*x + ky*y))


def beam_profile_gauss(x, y, r_factor):
    """
    Compute an apodized gaussian on the given x & y coordinates.
    """
    r_sq = x ** 2 + y ** 2
    return torch.exp(-(r_factor ** 2 * r_sq)) * (r_sq <= 1)


def build_square_k_space(k_min, k_max):
    """
    Constructs the k-space by creating a set of (k_x, k_y) coordinates.
    Fills the k_left and k_right matrices with the same k-space. (k_x, k_y) denote the k-space coordinates of the whole
    pupil. Only half SLM (and thus pupil) is modulated at a time, hence k_y (axis=1) must make steps of 2.

    Returns:
        k_space (np.ndarray): A 2xN array of k-space coordinates.
    """
    # Generate kx and ky coordinates
    kx_angles = np.arange(k_min, k_max + 1, 1)
    k_angles_min_even = (k_min if k_min % 2 == 0 else k_min + 1)        # Must be even
    ky_angles = np.arange(k_angles_min_even, k_max + 1, 2)              # Steps of 2

    # Combine kx and ky coordinates into pairs
    k_x = np.repeat(np.array(kx_angles)[np.newaxis, :], len(ky_angles), axis=0).flatten()
    k_y = np.repeat(np.array(ky_angles)[:, np.newaxis], len(kx_angles), axis=1).flatten()
    k_space = np.vstack((k_x, k_y))
    return k_space


def compute_tilt_mode(mode_shape, k, r_factor, ax=None, ay=None, x_min=-1, x_max=0, y_min=-1, y_max=1, ignore_warp=False,
                      ignore_amplitude=False):
    # Coordinates
    x = torch.linspace(x_min, x_max, mode_shape[1] // 2).view(1, -1, 1, 1)  # Normalized x coords
    y = torch.linspace(y_min, y_max, mode_shape[0]).view(-1, 1, 1, 1)  # Normalized y coords

    # Amplitude
    if ignore_amplitude:
        amplitude = 1
    else:
        amplitude = beam_profile_gauss(x, y, r_factor)

    # Phase
    if ignore_warp:
        phase_factor = tilt(x, y, k[1], k[0])
    else:
        wx, wy = warp_func_tilt(x, y, ax, ay)
        phase_factor = tilt(wx, wy, k[1], k[0])
    return amplitude * phase_factor


def compute_gram_tilt(r_factor, ax, ay, shape, k_min, k_max):
    # Initializations
    k_space = build_square_k_space(k_min=k_min, k_max=k_max)
    overlaps = torch.zeros(([k_space.shape[1]] * 2), dtype=torch.complex128)
    tilt_corrs = torch.zeros([k_space.shape[1]], dtype=torch.complex128)

    # Compute overlaps
    for i1, k1 in enumerate(k_space.T):
        warped_mode1 = compute_tilt_mode(shape, k1, r_factor, ax, ay)

        # plot_field(warped_mode1.squeeze().detach(), 1)
        # plt.title('Mode')
        # complex_colorbar(1)
        # plt.pause(0.1)

        for i2, k2 in enumerate(k_space.T):
            warped_mode2 = compute_tilt_mode(shape, k2, r_factor, ax, ay)
            overlaps[i1, i2] = field_correlation(warped_mode1, warped_mode2, dim=(0, 1, 2, 3))

        tilt_mode1 = compute_tilt_mode(shape, k1, r_factor, ignore_warp=True)
        tilt_corrs[i1] = field_correlation(warped_mode1, tilt_mode1, dim=(0, 1, 2, 3))

    return overlaps, tilt_corrs


def compute_similarity_from_corrs(corrs):
    return corrs.abs().pow(2).sum() / corrs.shape[0]

def warp_func_tilt(x, y, a, b, pow_factor = 2):
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
    wx = (a * x * x_to_powers * y_to_powers).sum(dim=(2, 3))
    wy = (b * y * x_to_powers * y_to_powers).sum(dim=(2, 3))

    return wx, wy
