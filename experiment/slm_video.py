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
from openwfs.algorithms import CustomBlindDualReference
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

size = (300, 300)

# Import variables
print('\nStart import modes...')
with h5py.File(phases_filepath, 'r') as f:
    phases_pw_half = f['init_phases_hr'][:, :, :, 0, 0]
    phases_ortho_pw_half = f['new_phases_hr'][:, :, :, 0, 0]

# Construct full pupil phase patterns from half pupil phases
mask_shape = phases_pw_half.shape[0:2]
phases_pw = np.concatenate((phases_pw_half, np.zeros(shape=phases_pw_half.shape)), axis=1)
phases_ortho_pw = np.concatenate((phases_ortho_pw_half, np.zeros(shape=phases_pw_half.shape)), axis=1)
split_mask = np.concatenate((np.zeros(shape=mask_shape), np.ones(shape=mask_shape)), axis=1)


# WFS arguments
if do_quick_test:
    algorithm_kwargs = {'phases': (phases_ortho_pw[:, :, 0:25], np.flip(phases_ortho_pw[:, :, 0:25]))}
    algorithm_common_kwargs = {'iterations': 2, 'phase_steps': 4, 'set1_mask': split_mask, 'do_try_full_patterns': True,
                               'progress_bar_kwargs': {'ncols': 60, 'leave': False}}
else:
    algorithm_kwargs = {'phases': (phases_ortho_pw, np.flip(phases_ortho_pw))}
    algorithm_common_kwargs = {'iterations': 6, 'phase_steps': 16, 'set1_mask': split_mask, 'do_try_full_patterns': True,
                               'progress_bar_kwargs': {'ncols': 60, 'leave': False}}

# Setup WFS sim
t = np.random.normal(size=size) + 1j * np.random.normal(size=size)
sim = SimulatedWFS(t=t)
reader = SLMPatternSaver(source=sim, slm=sim.slm, output_filepath=output_filepath)  # Saves SLM pattern at every fetch

# Run WFS
print('Run WFS simulation...')
alg = CustomBlindDualReference(feedback=reader, slm=sim.slm, slm_shape=size, **algorithm_common_kwargs, **algorithm_kwargs)
alg.execute()
