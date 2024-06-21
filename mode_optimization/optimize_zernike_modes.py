import torch
import numpy as np
import matplotlib.pyplot as plt
from zernike import RZern

from mode_functions import compute_gram

cart = RZern(6)
L, K = 10, 10
ddx = np.linspace(-1.0, 1.0, K)
ddy = np.linspace(-1.0, 1.0, L)
xv, yv = np.meshgrid(ddx, ddy)
cart.make_cart_grid(xv, yv, unit_circle=False)

num_modes = 11
modes = torch.zeros((L, K, num_modes), dtype=torch.complex64)
phi_amp = 2*np.pi

for i in range(0, num_modes):
    c = np.zeros(cart.nk)
    c[i] = 1.0
    Phi = torch.tensor(cart.eval_grid(c, matrix=True))
    modes[:, :, i] = torch.exp(1j * phi_amp * Phi)

gram = compute_gram(modes) / (L*K)

plt.figure()
plt.imshow(np.abs(gram))
plt.colorbar()
plt.show()
