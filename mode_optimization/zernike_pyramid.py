import torch
import numpy as np
import matplotlib.pyplot as plt

from zernike import zernike_cart, zernike_order
from helper_functions import plot_field


def zernike_field(x, y, n, m, phase_coeff):
    """Compute the field of a Zernike mode."""
    amplitude = x**2 + y**2 <= 1                # Unit circle
    phase = phase_coeff * zernike_cart(x, y, n, m)
    return amplitude * torch.exp(1j * phase)


# Settings
phase_coeff = np.pi
j_max = 16
grid_num = 5

# Coordinates
x = torch.linspace(-1, 1, 80).view(1, -1)
y = torch.linspace(-1, 1, 80).view(-1, 1)

# Prepare plot
plt.figure(figsize=(10, 9), dpi=80)
plt.tight_layout()
plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)

# Loop over modes
for j in range(1, j_max):
    n, m = zernike_order(j)
    field = zernike_field(x, y, n, m, phase_coeff=phase_coeff)
    plt.subplot(grid_num, grid_num, int(n * grid_num + m+1))
    plot_field(field, scale=1)
    plt.title(f'j: {j}, n: {int(n)}, m: {int(m)}')
    plt.xticks([])
    plt.yticks([])

plt.show()
