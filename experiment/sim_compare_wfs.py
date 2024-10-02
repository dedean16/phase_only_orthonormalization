"""
Simulate and compare three different basis settings in a wavefront shaping algorithm.
"""
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt

from openwfs.algorithms import DualReference
from openwfs.simulation import SimulatedWFS

from phase_only_orthonormalization.directories import localdata


phases_filepath = os.path.join(localdata, 'ortho-plane-waves.hdf5')

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
algorithm_kwargs = [
    {'phase_patterns': phase_patterns_pw, 'amplitude': 'uniform'},
    {'phase_patterns': phase_patterns_pw, 'amplitude': amplitude},
    {'phase_patterns': phase_patterns_ortho_pw,
     'amplitude': amplitude}
]
algorithm_common_kwargs = {'iterations': 2, 'phase_steps': 6, 'group_mask': group_mask}

# ===== Prepare simulation ===== #
size = full_beam_amplitude.shape
t = np.random.normal(size=size) + 1j * np.random.normal(size=size)
sim = SimulatedWFS(t=t, beam_amplitude=full_beam_amplitude)
slm = sim.slm

# ===== Loop and simulate ===== #
for n_alg, alg_kwargs in enumerate(algorithm_kwargs):
    alg = DualReference(feedback=sim, slm=slm, **algorithm_common_kwargs, **alg_kwargs)
    result = alg.execute()

    # Measure intensity with flat and shaped wavefronts
    slm.set_phases(0)
    before = sim.read()
    slm.set_phases(-np.angle(result.t))
    after = sim.read()
    print(f"Intensity in the target increased from {before:.3g} to {after:.3g} (factor of {after/before:.3g}x)")
