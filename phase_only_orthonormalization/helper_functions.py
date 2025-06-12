from typing import Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.axes import Axes
import git
import h5py


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
        imshow_kwargs: Keyword arguments for matplotlib's imshow.
    """
    rgb = complex_to_rgb(array, scale)
    plt.imshow(rgb, **imshow_kwargs)


def plot_scatter_field(x, y, array, scale, scatter_kwargs=None):
    """
    Plot complex scattered data as RGB values.
    """
    if scatter_kwargs is None:
        scatter_kwargs = {'s': 80}
    rgb = complex_to_rgb(array, scale, axis=1)
    plt.scatter(x, y, c=rgb, **scatter_kwargs)


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


def complex_colorwheel(ax: Axes = None, shape: Tuple[int, int] = (100, 100), imshow_kwargs: dict = {},
                       arrow_props: dict = {}, text_kwargs: dict = {}, amplitude_str: str = 'A',
                       phase_str: str = '$\\phi$'):
    """
    Create an rgb image for a colorwheel representing the complex unit circle.

    Args:
        ax: Matplotlib Axes.
        shape: Number of pixels in each dimension.
        imshow_kwargs: Keyword arguments for matplotlib's imshow.
        arrow_props: Keyword arguments for the arrows.
        text_kwargs: Keyword arguments for the text labels.
        amplitude_str: Text label for the amplitude arrow.
        phase_str: Text label for the phase arrow.

    Returns:
        rgb_wheel: rgb image of the colorwheel.
    """
    if ax is None:
        ax = plt.gca()

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
    ax.text(**{'x': 0.15, 'y': 0.55, 's': phase_str, 'color': 'white', 'fontsize': 15, 'ha': 'center', 'va': 'center',
               **text_kwargs})

    # Hide axes spines and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def grid_bitmap(x, y, domain, grid_length, line_width):
    """
    Create a bitmap image of a grid. The grid lines are set to 1. The grid cells interior to 0.

    Args:
        x: Input coordinate x.
        y: Input coordinate y.
        domain: Dictionary containing domain limits.
        grid_length: Length of one grid cell.
        line_width: Width of one grid line.

    Returns:
        Bitmap image of a grid.
    """
    x_map = (x % grid_length) < line_width
    y_map = (y % grid_length) < line_width
    in_domain = (x >= domain['x_min']) & (x <= domain['x_max']) & (y >= domain['y_min']) & (y <= domain['y_max'])
    return (x_map | y_map) & in_domain


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


def add_dict_as_hdf5group(name: str, dic: dict, hdf: h5py.File | h5py.Group):
    """
    Add python dictionary as group to HDF5 file. Supports nested dictionaries.

    Args:
        name: Name of the group
        dic: Dictionary to add.
        hdf: HDF5 file or group to add to.
    """
    subgroup = hdf.create_group(name)
    for key, value in dic.items():
        if isinstance(value, dict):
            add_dict_as_hdf5group(name=key, dic=value, hdf=subgroup)
        else:
            subgroup.create_dataset(key, data=value)


def add_dict_sequence_as_hdf5_groups(name: str, seq: list | tuple, hdf: h5py.File | h5py.Group):
    """
    Add list or tuple of dictionaries as group to HDF5 file. The list/tuple items will be stored as subgroups,
    with the index numbers as keys. Supports nested dictionaries. The list/tuple may also contain items of other type,
    as long as they are supported by h5py.

    Args:
        name: Name of the list or tuple group.
        seq: List or tuple of dictionaries to add.
        hdf: HDF5 file or group to add to.
    """
    subgroup = hdf.create_group(name)
    for n, item in enumerate(seq):
        if isinstance(item, dict):
            add_dict_as_hdf5group(name=f'{n}', dic=item, hdf=subgroup)
        else:
            subgroup.create_dataset(name=f'{n}', data=item)


def get_dict_from_hdf5(group: h5py.Group) -> dict:
    """
    Retrieve a hdf5 file or group as a dictionary. Supports nested dictionaries. Can be used on the main group as well
    to get a dictionary of the whole hdf5 structure.

    Args:
        group: hdf5 group.

    Returns: the group as dictionary
    """
    dic = {}
    for key in group.keys():
        if group[key].__class__.__name__ == 'Group':
            dic[key] = get_dict_from_hdf5(group[key])
        else:
            dic[key] = group[key][()]
    return dic


def scalar_factor_least_squares(x, y):
    """
    Compute least squares solution to: y = bx, solve for b.

    Args:
        x, y: Arrays containing the data points.

    Returns: The least squares solution b.
    """
    A = np.vstack([x]).T
    b = np.linalg.lstsq(A, y)[0]
    return b


def place_xy_arrows(ax, arrow_props={'head_width': 0.1, 'head_length': 0.15, 'fc': 'black', 'ec': 'black'},
                    label_props={'fontsize': 14, 'ha': 'center', 'va': 'center'}):
    # Draw arrows
    ax.arrow(0, 0, 1, 0, **arrow_props)
    ax.arrow(0, 0, 0, 1, **arrow_props)

    # Set the limits of the plot
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)

    # Add labels for x and y axes
    ax.text(1.05, 0.2, 'x', **label_props)
    ax.text(0.2, 1.05, 'y', **label_props)

    # Customize the plot appearance
    ax.set_aspect('equal')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])
