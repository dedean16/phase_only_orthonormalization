import torch
import numpy as np

from mode_functions import gaussian, warp_func, field_correlation
from helper_functions import build_square_k_space


def tilt(x, y, kx, ky):
    return torch.exp(1j * np.pi * (kx*x + ky*y))


def compute_tilt_mode(mode_shape, k, r_factor, ax=None, ay=None, x_min=-1, x_max=0, y_min=-1, y_max=1, ignore_warp=False,
                      ignore_amplitude=False):
    # Coordinates
    x = torch.linspace(x_min, x_max, mode_shape[1] // 2).view(1, -1, 1, 1)  # Normalized x coords
    y = torch.linspace(y_min, y_max, mode_shape[0]).view(-1, 1, 1, 1)  # Normalized y coords

    # Amplitude
    if ignore_amplitude:
        amplitude = 1
    else:
        amplitude = gaussian(x, y, r_factor)

    # Phase
    if ignore_warp:
        phase_factor = tilt(x, y, k[1], k[0])
    else:
        wx, wy = warp_func(x, y, ax, ay)
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

