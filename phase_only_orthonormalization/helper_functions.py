from typing import Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.axes import Axes
import git


def slope_step(a, width=0.1):
    """
    A sloped step function from 0 to 1.

    Args:
        a: Input array
        width: width of the sloped step.

    Returns:
        An array the size of a, with the result of the sloped step function.
    """
    return (a >= width) + a/width * (0 < a) * (a < width)


def linear_blend(a, b, blend):
    """Return a linear, element-wise blend between two arrays a and b."""
    return a*blend + b*(1-blend)


def complex_to_rgb(array, scale, axis=2):
    """Generate RGB values to represent values of a complex array."""
    h = np.expand_dims(np.angle(array) / (2 * np.pi) + 0.5, axis=axis)
    s = np.ones_like(h)
    v = np.expand_dims(np.abs(array) * scale, axis=axis).clip(min=0, max=1)
    hsv = np.concatenate((h, s, v), axis=axis)
    rgb = hsv_to_rgb(hsv)
    return rgb


def plot_field(array, scale, imshow_kwargs={}):
    """
    Plot a complex array as an RGB image.

    The phase is represented by the hue, and the magnitude by the value, i.e. black = zero, brightness shows amplitude,
    and the colors represent the phase.

    Args:
        array(ndarray): complex array to be plotted.
        scale(float): scaling factor for the magnitude. The final value is clipped to the range [0, 1].
    """
    rgb = complex_to_rgb(array, scale)
    plt.imshow(rgb, **imshow_kwargs)
    # plt.set_cmap('hsv')


def plot_scatter_field(x, y, array, scale, scatter_kwargs=None):
    """
    Plot complex scattered data as RGB values.
    """
    if scatter_kwargs is None:
        scatter_kwargs = {'s': 80}
    rgb = complex_to_rgb(array, scale, axis=1)
    plt.scatter(x, y, c=rgb, **scatter_kwargs)
    # plt.set_cmap('hsv')


def complex_colorbar(scale, width_inverse: int = 15):
    """
    Create an rgb colorbar for complex numbers and return its Axes handle.
    """
    amp = np.linspace(0, 1.01, 10).reshape((1, -1))
    phase = np.linspace(0, 249 / 250 * 2 * np.pi, 250).reshape(-1, 1) - np.pi
    z = amp * np.exp(1j * phase)
    rgb = complex_to_rgb(z, 1)
    ax = plt.subplot(1, width_inverse, width_inverse)
    plt.imshow(rgb, aspect='auto', extent=(0, scale, -np.pi, np.pi))

    # Ticks and labels
    ax.set_yticks((-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi), ('$-\\pi$', '$-\\pi/2$', '0', '$\\pi/2$', '$\\pi$'))
    ax.set_xlabel('amp.')
    ax.set_ylabel('phase (rad)')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    return ax


def complex_colorwheel(ax: Axes = plt.gca(), shape: Tuple[int, int] = (100, 100), imshow_kwargs={},
                       arrow_props={}, text_kwargs={}, amplitude_str='A', phase_str='$\\phi$'):
    """
    Create an rgb image for a colorwheel representing the complex unit circle.

    Returns:
        rgb_wheel: rgb image of the colorwheel.
    """
    x = np.linspace(-1, 1, shape[1]).reshape(1, -1)
    y = np.linspace(-1, 1, shape[0]).reshape(-1, 1)
    z = x + 1j*y
    rgb = complex_to_rgb(z, scale=1)
    step_width = 1.5 / shape[1]
    blend = np.expand_dims(slope_step(1 - np.abs(z) - step_width, width=step_width), axis=2)
    rgba_wheel = np.concatenate((rgb, blend), axis=2)
    ax.imshow(rgba_wheel, extent=(-1, 1, -1, 1), **imshow_kwargs)

    # Add arrows with annotations
    ax.annotate('', xy=(-0.98/np.sqrt(2),)*2, xytext=(0, 0), arrowprops={'color': 'white', 'width': 1.8,
        'headwidth': 5.0, 'headlength': 6.0, **arrow_props})
    ax.text(**{'x': -0.4, 'y': -0.8, 's': amplitude_str, 'color': 'white', 'fontsize': 15, **text_kwargs})
    ax.annotate('', xy=(0, 0.9), xytext=(0.9, 0),
                arrowprops={'connectionstyle': 'arc3,rad=0.4', 'color': 'white', 'width': 1.8, 'headwidth': 5.0,
                            'headlength': 6.0, **arrow_props})
    ax.text(**{'x': 0.1, 'y': 0.5, 's': phase_str, 'color': 'white', 'fontsize': 15, **text_kwargs})

    # Hide axes spines and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def grid_bitmap(x, y, grid_length, line_width):
    """
    Create a bitmap image of a grid. The grid lines are set to 1. The grid cells interior to 0.

    Args:
        x: Input coordinate x.
        y: Input coordinate y.
        grid_length: Length of one grid cell.
        line_width: Width of one grid line.

    Returns:
        Bitmap image of a grid.
    """
    x_map = (x % grid_length) > line_width
    y_map = (y % grid_length) > line_width
    return 1 - x_map * y_map


def gitinfo() -> dict:
    """
    Return a dict with info about the current git commit and repository.
    """
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    working_dir = repo.working_dir
    diff = repo.git.diff()
    commit_timestamp = repo.head.object.committed_datetime.timestamp()
    git_info = {'sha': sha, 'working_dir': working_dir, 'diff': diff, 'commit_timestamp': commit_timestamp}
    return git_info


def n_choose_k(n, k):
    """
    N choose k. Also known as comb in scipy.
    """
    if n is not torch.Tensor:
        n = torch.tensor(n)
    if k is not torch.Tensor:
        k = torch.tensor(k)
    return (torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma((n - k) + 1)).exp()


def factorial(x):
    if x is not torch.Tensor:
        x = torch.tensor(x)
    return torch.exp(torch.lgamma(x+1))


def t_abs(x):
    if x is not torch.Tensor:
        x = torch.tensor(x)
    return x.abs()


def mse(a, b, dim=None):
    """Mean Squared |Error|"""
    return (a-b).abs().pow(2).mean(dim=dim)
