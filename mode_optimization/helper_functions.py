import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import git


def complex_to_rgb(array, scale, axis=2):
    """Generate RGB values to represent values of a complex array."""
    h = np.expand_dims(np.angle(array) / (2 * np.pi) + 0.5, axis=axis)
    s = np.ones_like(h)
    v = np.expand_dims(np.abs(array) * scale, axis=axis).clip(min=0, max=1)
    hsv = np.concatenate((h, s, v), axis=axis)
    rgb = hsv_to_rgb(hsv)
    return rgb


def plot_field(array, scale):
    """
    Plot a complex array as an RGB image.

    The phase is represented by the hue, and the magnitude by the value, i.e. black = zero, brightness shows amplitude,
    and the colors represent the phase.

    Args:
        array(ndarray): complex array to be plotted.
        scale(float): scaling factor for the magnitude. The final value is clipped to the range [0, 1].
    """
    rgb = complex_to_rgb(array, scale)
    plt.imshow(rgb)
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
