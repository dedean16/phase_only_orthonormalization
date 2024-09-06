import numpy as np
from numpy import ndarray as nd
import matplotlib.pyplot as plt

from openwfs.core import Processor
from openwfs.utilities import project
from openwfs.algorithms.utilities import WFSResult


class OffsetRemover(Processor):
    """Processor that removes offset."""
    def __init__(self, source, offset, dtype_out='float64'):
        super().__init__(source, multi_threaded=False)
        self.offset = offset
        self.source = source
        self.dtype_out = dtype_out

    def _fetch(self, source_data: np.ndarray) -> np.ndarray:
        return source_data.astype(self.dtype_out) - self.offset


class RandomSLMShutter:
    """Create a shutter by putting a random pattern on the SLM."""
    def __init__(self, slm):
        self.slm = slm
        self._open = False

    @property
    def open(self):
        return self._open

    @open.setter
    def open(self, value):
        if value:
            self.slm.set_phases(0)
        else:
            self.slm.set_phases(np.random.rand(300, 300) * 2*np.pi)
        self._open = value


class NoWFS:
    """
    WFS algorithm that doesn't do anything, and returns a transmission matrix of 1.0.
    """
    def __init__(self, feedback, slm):
        self.feedback = feedback
        self.slm = slm

    def execute(self):
        return WFSResult(t=np.asarray((1.0,)), t_f=np.asarray((1.0, 0.0, 0.0)),
                         axis=0, fidelity_amplitude=0, fidelity_noise=0, fidelity_calibration=0, n=0)
