"""
This script is used for finding settings for plotting complex fields using the oklab colorspace. The ideal settings must
produce values that fit within the sRGB colour gamut.

Checks also if settings produce colours that fit in a CMYK gamut, used by printers.
Note that there are many CMYK colour spaces. This one uses the default from the colour-science package.
"""
import numpy as np
import matplotlib.pyplot as plt
# import colour
from tqdm import tqdm

from helper_functions import complex_to_rgb




Nxy = 100
NL = 100
Nab = 100

x = np.linspace(-1.0, 1.0, Nxy).reshape((1, Nxy))
y = np.linspace(-1.0, 1.0, Nxy).reshape((Nxy, 1))
z_uc = x + 1j*y
z = z_uc * (np.abs(z_uc) < 1.0)

Lmax_range = np.linspace(0.01, 1.0, NL)
# abmax_range = np.linspace(0.001, 0.25, Nab)
abmax_range = np.linspace(0.001, 0.25, Nab)

out_of_bounds_sRGB = np.zeros((NL, Nab))
out_of_bounds_CMYK = np.zeros((NL, Nab))

# plt.figure()

for iL in tqdm(range(NL)):
    for iab in range(Nab):
        srgb = complex_to_rgb(z, scale=1.0, Lmax=Lmax_range[iL], abmax=abmax_range[iab], colorspace='oklab')
        out_of_bounds_sRGB[iL, iab] = np.sum(srgb < 0.0) + np.sum(srgb > 1.0)

        # plt.clf()
        # plt.imshow((srgb*255).clip(0.0, 255.0).astype(np.uint8))
        # plt.title(f'Lmax={Lmax_range[iL]:.2f}, abmax={abmax_range[iab]:.2f}')
        # plt.pause(0.05)

        # cmyk = colour.CMY_to_CMYK(colour.RGB_to_CMY(srgb))
        # out_of_bounds_CMYK[iL, iab] = np.sum(cmyk < 0.0) + np.sum(cmyk > 1.0)

# plt.imshow(srgb, extent=(-1, 1, -1, 1))
# plt.imshow(np.abs(z))
plt.figure(figsize=(4.5, 7))
plt.imshow(out_of_bounds_sRGB, extent=(abmax_range.min(), abmax_range.max(), Lmax_range.min(), Lmax_range.max()), origin='lower')
plt.xlabel('ab max')
plt.ylabel('L max')
plt.title('how many outside sRGB gamut')
plt.gca().set_aspect('auto')
plt.colorbar()

plt.figure(figsize=(4.5, 7))
plt.imshow(out_of_bounds_sRGB > 0.0, extent=(abmax_range.min(), abmax_range.max(), Lmax_range.min(), Lmax_range.max()), origin='lower')
plt.xlabel('ab max')
plt.ylabel('L max')
plt.title('any outside sRGB gamut')
plt.gca().set_aspect('auto')
plt.colorbar()

# plt.figure(figsize=(4.5, 7))
# plt.imshow(out_of_bounds_CMYK, extent=(abmax_range.min(), abmax_range.max(), Lmax_range.min(), Lmax_range.max()), origin='lower')
# plt.xlabel('ab max')
# plt.ylabel('L max')
# plt.title('how many outside CMYK gamut')
# plt.gca().set_aspect('auto')
# plt.colorbar()
#
# plt.figure(figsize=(4.5, 7))
# plt.imshow(out_of_bounds_CMYK > 0.0, extent=(abmax_range.min(), abmax_range.max(), Lmax_range.min(), Lmax_range.max()), origin='lower')
# plt.xlabel('ab max')
# plt.ylabel('L max')
# plt.title('any outside CMYK gamut')
# plt.gca().set_aspect('auto')
# plt.colorbar()

plt.show()
