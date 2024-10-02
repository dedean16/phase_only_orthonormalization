Phase-only orthonormalization
=============================

Orthonormalization of phase-only 2D basis functions using PyTorch. Read the pre-print research paper at https://arxiv.org/abs/2409.04565

Installation
------------
This repository uses Poetry for installing its dependencies.
Please see https://python-poetry.org/.

When Poetry is installed, dependencies can be installed by running poetry inside the repository's directory.

Install with the default PyTorch version for your platform:
 - Platform default version: ``poetry install --with torchdefault,dev,experiment``

Note 1: The ``dev`` group is recommended to run the unit tests.

Note 2: If you want to install a different version of PyTorch, exclude the ``torchdefault`` group and
install the desired ``torch`` version separately. Please see https://pytorch.org/ for installation instructions.

Note 3: The ``experiment`` group is required for running and analyzing the wavefront shaping experiment
that uses the orthonormalized basis.


Running the scripts
-------------------
The orthonormalization scripts can be found in the ``phase_only_orthonormalization`` directory.
After installation is complete, run the scripts
``optimize_plane_wave_modes.py``, ``optimize_zernike_modes.py``, ``optimize_laguerre_gaussian_modes.py``
or ``optimize_polar_harmonic_modes.py`` to orthonormalize a basis based on plane waves, zernike polynomials,
Laguerre Gauss modes or polar harmonics respectively.
Options for plotting and/or saving are available near the top of each script.

Scripts for running and analyzing the wavefront shaping experiments can be found in the ``experiment`` directory.
Ensure the ``experiment`` poetry group is installed. ``batch_compare_wfs.py`` is the measurement script for comparing
wavefront shaping performance using the different bases. Note that running this script requires the corresponding
hardware. ``analyze_compare_wfs.py`` may be used to analyze the measurement results. Lastly, ``sim_slm_video.py`` may be
used to simulate a wavefront shaping experiment and export the SLM frames.

Before running any script, please ensure the variables defined in ``phase_only_orthonormalization/directories.py``
point to valid directories. Additionally, for the scripts that import/export data, please ensure that (sub)directories
and file paths in the settings (near the top of the script) are valid.
