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
from experiment_helper_classes import NoWFS


# === Settings === #
# Note: WFS settings further down
phases_filepath = os.path.join(localdata, 'ortho-plane-waves-80x40.hdf5')

runs_per_noise_level = 12
one_over_noise_range = np.asarray([0.05, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 7.0, 10.0, 13.0, 16.0, 20.0]) / 1.5

# Import variables
print('\nStart import modes...')
with h5py.File(phases_filepath, 'r') as f:
    phases_pw_half = f['init_phases_hr'][:, :, :, 0, 0]
    phases_ortho_pw_half = f['new_phases_hr'][:, :, :, 0, 0]
    amplitude_half = f['amplitude_profile'][:, :, 0, 0, 0]

# ===== Construct full SLM modes for Dual Reference ===== #
N1, N2, M = phases_pw_half.shape

# Expand half-SLM-modes to full SLM (second half are zeros)
phases_pw = np.concatenate((phases_pw_half, np.zeros(shape=(N1, N2, M))), axis=1)
phases_ortho_pw = np.concatenate((phases_ortho_pw_half, np.zeros(shape=(N1, N2, M))), axis=1)
full_beam_amplitude_unnorm = np.concatenate((amplitude_half, np.flip(amplitude_half)), axis=1)
full_beam_amplitude = full_beam_amplitude_unnorm / np.sqrt((full_beam_amplitude_unnorm**2).sum())

# Phases and amplitude of both groups, both halves
phase_patterns_pw = (phases_pw, np.flip(phases_pw))
phase_patterns_ortho_pw = (phases_ortho_pw, np.flip(phases_ortho_pw))
amplitude = (full_beam_amplitude, full_beam_amplitude)

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
algorithm_common_kwargs = {'iterations': 4, 'phase_steps': 6, 'group_mask': group_mask}

# ===== Prepare simulation ===== #
size = full_beam_amplitude.shape

# === Prepare result arrays === #
flat_signals = np.zeros((runs_per_noise_level, len(one_over_noise_range), len(algorithm_kwargs)))
shaped_signals = flat_signals.copy()

# ===== Loop and simulate ===== #
print('Start simulations...')
progress_bar = tqdm(total=flat_signals.size)

for r in range(runs_per_noise_level):
    t = np.random.normal(size=size) + 1j * np.random.normal(size=size)
    sim = SimulatedWFS(t=t, beam_amplitude=full_beam_amplitude)
    slm = sim.slm

    for n, one_over_noise in enumerate(one_over_noise_range):
        noise_level = 1 / one_over_noise
        noisy_detect = GaussianNoise(source=sim, std=noise_level)

        for a, alg_kwargs in enumerate(algorithm_kwargs):
            # alg = DualReference(feedback=noisy_detect, slm=slm, **algorithm_common_kwargs, **alg_kwargs)
            alg = NoWFS(feedback=noisy_detect, slm=slm)
            result = alg.execute()

            # Intensity with flat wavefront
            slm.set_phases(0)
            before = sim.read()
            flat_signals[r, n, a] = before

            # Intensity with shaped wavefront
            slm.set_phases(-np.angle(result.t))
            after = sim.read()
            shaped_signals[r, n, a] = after

            progress_bar.update()


mean_background_signal = flat_signals[:, 0, 0].mean()
enhancement_means = shaped_signals.mean(axis=0) / mean_background_signal
enhancement_stds = shaped_signals.std(axis=0) / mean_background_signal
mean_initial_snr_range = mean_background_signal * one_over_noise_range

print(f'\nBackground signal: {mean_background_signal:.2f} ± {flat_signals[:, 0, 0].std():.2f}')

for a, alg_label in enumerate(alg_labels):
    plt.errorbar(mean_initial_snr_range, enhancement_means[:, a], enhancement_stds[:, a], label=alg_label)

plt.xlabel('Mean Initial SNR')
plt.ylabel('Enhancement $\\eta$')
plt.xlim((0, None))
plt.ylim((0, None))
plt.legend()
plt.title(f'WFS performance\n{runs_per_noise_level} runs per noise level')
plt.show()
