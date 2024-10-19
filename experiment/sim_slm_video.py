"""
Create video frames showing the SLM during a wavefront shaping simulation using the orthonormalized plane wave basis.
Before running this script, ensure the paths in directories.py and the file paths defined in the save settings below are
valid.

Please check import and output file paths before running this script. This script creates PNG files in the specified
output folder (if it exists). These can be converted to a video with e.g. ffmpeg:
ffmpeg -framerate 30 -i slm-pattern_%05d.png -c:v libx265 -preset slow -crf 22 slm-patterns.mp4
"""
# Built-in
import os

# External (3rd party)
import numpy as np
import h5py

# External (ours)
from openwfs.algorithms import DualReference
from openwfs.simulation import SimulatedWFS

# Internal
from experiment_helper_classes import SLMPatternSaver
from phase_only_orthonormalization.directories import localdata


# ========== Settings ========== #
do_quick_test = False       # False: Full run, True: Quick test run with a few modes

# Saving
output_filepath = os.path.join(localdata, "slm-frames/slm-pattern")

# Import modes
phases_filepath = os.path.join(localdata, "ortho-plane-waves-hires.hdf5")

size = (400, 400)

# Import variables
print('\nStart import modes...')
with h5py.File(phases_filepath, 'r') as f:
    phases_pw_half = f['init_phases_hr'][:, :, :, 0, 0].transpose(2, 0, 1)
    phases_ortho_pw_half = f['new_phases_hr'][:, :, :, 0, 0].transpose(2, 0, 1)
    amplitude_half = f['amplitude_profile'][:, :, 0, 0, 0]

# Construct full SLM basis array from loaded half-SLM basis
mask_shape = phases_pw_half.shape[1:3]
phases_pw = np.concatenate((phases_pw_half, np.zeros(shape=phases_pw_half.shape)), axis=2)
phases_ortho_pw = np.concatenate((phases_ortho_pw_half, np.zeros(shape=phases_pw_half.shape)), axis=2)
split_mask = np.concatenate((np.zeros(shape=mask_shape), np.ones(shape=mask_shape)), axis=1)
full_beam_amplitude_unnorm = np.concatenate((amplitude_half, np.flip(amplitude_half)), axis=1)
full_beam_amplitude = full_beam_amplitude_unnorm / np.sqrt((full_beam_amplitude_unnorm ** 2).sum())
uniform_amplitude = 2 * np.ones_like(full_beam_amplitude) / full_beam_amplitude.size

# Phases and amplitude of both groups, both halves
phase_patterns_pw = (phases_pw, np.flip(phases_pw))
phase_patterns_ortho_pw = (phases_ortho_pw, np.flip(phases_ortho_pw))

# WFS arguments
if do_quick_test:
    algorithm_kwargs = {'phase_patterns': (phases_ortho_pw[:, :, 0:20], np.flip(phases_ortho_pw[:, :, 0:20]))}
    algorithm_common_kwargs = {'iterations': 2, 'phase_steps': 4, 'group_mask': split_mask, 'amplitude': full_beam_amplitude}
else:
    algorithm_kwargs = {'phase_patterns': (phases_ortho_pw, np.flip(phases_ortho_pw))}
    algorithm_common_kwargs = {'iterations': 3, 'phase_steps': 8, 'group_mask': split_mask, 'amplitude': full_beam_amplitude}

# Setup WFS sim
t = np.random.normal(size=size) + 1j * np.random.normal(size=size)
sim = SimulatedWFS(t=t)
reader = SLMPatternSaver(source=sim, slm=sim.slm, output_filepath=output_filepath)  # Saves SLM pattern at every fetch

# Run WFS
print(f'Run WFS simulation... Saving frames to {output_filepath}')
alg = DualReference(feedback=reader, slm=sim.slm, **algorithm_common_kwargs, **algorithm_kwargs)
alg.execute()
