Mode Optimization
=================

Orthogonalization of phase-only 2D basis functions using PyTorch.

Installation
------------
This repository uses Poetry for installing its dependencies.
Please see https://python-poetry.org/.

When Poetry is installed, dependencies can be installed by running poetry
inside the repository's directory.

Choose the PyTorch version to install:
 - Platform default: ``poetry install --with default,dev``
 - With CUDA12: ``poetry install --with cuda12,dev``

Note 1: The ``dev`` group is recommended to run the unit tests.

Note 2: If you want to install a different version of PyTorch,
e.g. the CUDA11 version, exclude the ``default`` and ``cuda12`` group
(i.e. run ``poetry install`` or ``poetry install --with dev``) and
install the desired ``torch`` version separately.
