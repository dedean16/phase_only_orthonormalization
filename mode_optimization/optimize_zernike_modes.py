import torch
import matplotlib.pyplot as plt

from zernike import zernfun_cart


x = torch.linspace(-1, 1, 100).view(1, -1)
y = torch.linspace(-1, 1, 100).view(-1, 1)

z = zernfun_cart(x, y, 3, 1, circle=True)

plt.imshow(z)
plt.show()
