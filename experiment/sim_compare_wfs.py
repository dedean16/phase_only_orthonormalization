"""
Simulate and compare three different basis settings in a wavefront shaping algorithm.
"""
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

from openwfs.algorithms import DualReference
from openwfs.simulation import SimulatedWFS
from openwfs.simulation.mockdevices import GaussianNoise

from phase_only_orthonormalization.directories import localdata


# === Settings === #
# Note: WFS settings further down
phases_filepath = os.path.join(localdata, 'ortho-plane-waves.hdf5')

# runs_per_noise_level = 10
# gauss_noise_range = tuple(2 * x ** 2 for x in range(8))
runs_per_noise_level = 2
gauss_noise_range = tuple(2 * x ** 2 for x in range(2))

# Import variables
print('\nStart import modes...')
with h5py.File(phases_filepath, 'r') as f:
    modes_pw_half = f['init_modes'][()]
    modes_ortho_pw_half = f['new_modes'][()]

# ===== Construct full SLM modes for Dual Reference ===== #
N1, N2, M = modes_pw_half.shape

# Expand half-SLM-modes to full SLM (second half are zeros)
modes_pw = np.concatenate((modes_pw_half, np.zeros(shape=(N1, N2, M))), axis=1)
modes_ortho_pw = np.concatenate((modes_ortho_pw_half, np.zeros(shape=(N1, N2, M))), axis=1)

# Phases and amplitude of 1 group
phases_pw = np.angle(modes_pw)
phases_ortho_pw = np.angle(modes_ortho_pw)
amplitude_profile = abs(modes_ortho_pw[:, :, 0])

# Phases and amplitude of both groups, both halves
phase_patterns_pw = (phases_pw, np.flip(phases_pw))
phase_patterns_ortho_pw = (phases_ortho_pw, np.flip(phases_ortho_pw))
amplitude = (amplitude_profile, np.flip(amplitude_profile))
full_beam_amplitude = amplitude[0] + amplitude[1]

# Group mask
group_mask = np.concatenate((np.zeros((N1, N2)), np.ones((N1, N2))), axis=1)

# ===== WFS settings ===== #
alg_labels = ['PW uniform', 'PW trunc gauss', 'ortho PW trunc gauss']
algorithm_kwargs = [
    {'phase_patterns': phase_patterns_pw, 'amplitude': 'uniform'},
    {'phase_patterns': phase_patterns_pw, 'amplitude': amplitude},
    {'phase_patterns': phase_patterns_ortho_pw,
     'amplitude': amplitude}
]
# algorithm_common_kwargs = {'iterations': 4, 'phase_steps': 8, 'group_mask': group_mask}
algorithm_common_kwargs = {'iterations': 2, 'phase_steps': 4, 'group_mask': group_mask}

# ===== Prepare simulation ===== #
size = full_beam_amplitude.shape

# === Prepare result arrays === #
flat_signals = np.zeros((runs_per_noise_level, len(gauss_noise_range), len(algorithm_kwargs)))
shaped_signals = flat_signals.copy()

# ===== Loop and simulate ===== #
print('Start simulations...')
progress_bar = tqdm(total=flat_signals.size)

for r in range(runs_per_noise_level):
    t = np.random.normal(size=size) + 1j * np.random.normal(size=size)
    sim = SimulatedWFS(t=t, beam_amplitude=full_beam_amplitude)
    slm = sim.slm

    for n, noise_level in enumerate(gauss_noise_range):
        noisy_detect = GaussianNoise(source=sim, std=noise_level)

        for a, alg_kwargs in enumerate(algorithm_kwargs):
            alg = DualReference(feedback=noisy_detect, slm=slm, **algorithm_common_kwargs, **alg_kwargs)
            result = alg.execute()

            # Intensity with flat wavefront
            slm.set_phases(0)
            before = sim.read()
            flat_signals[r, n, a] = before

            # Intensity with shaped wavefront
            slm.set_phases(-np.angle(result.t))
            after = sim.read()
            shaped_signals[r, n, a] = after
            # print(f"Intensity in the target increased from {before:.3g} to {after:.3g} (factor of {after/before:.3g}x)")

            progress_bar.update()


mean_background_signal = flat_signals[:, 0, 0].mean()
enhancement_means = shaped_signals.mean(axis=0) / mean_background_signal
enhancement_stds = shaped_signals.std(axis=0) / mean_background_signal

for a, alg_label in enumerate(alg_labels):
    plt.errorbar(gauss_noise_range, enhancement_means[:, a], enhancement_stds[:, a], label=alg_label)

plt.xlabel('Gaussian noise std $\\sigma$')
plt.ylabel('Enhancement $\\eta$')
plt.legend()
plt.title('WFS performance')
plt.show()
