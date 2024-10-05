import numpy as np
from matplotlib.image import imsave

from openwfs.core import Processor, Detector
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
        return WFSResult(t=np.asarray((1.0,)), axis=0, fidelity_amplitude=0, fidelity_noise=0, fidelity_calibration=0, n=0)


class SLMPatternSaver(Processor):
    """
    Processor that saves SLM phase pattern every time its data is fetched.
    Then passes on its source to the next processor(s). The SLM patterns will be saved as PNG images.
    """
    def __init__(self, source, slm, output_filepath: str, cmap='gist_gray'):
        """
        Args:
            source: Source Detector/processor from which this object will receive data.
            slm: SLM object to save the state of.
            output_filepath: Output filepath to save SLM patterns to. A suffix counter will be added to the filepath.
        """
        super().__init__(source, multi_threaded=False)
        self.source = source
        self.slm = slm
        self.output_filepath = output_filepath
        self.counter = 0
        self.cmap = cmap

    def _fetch(self, source_data: np.ndarray) -> np.ndarray:
        # Save SLM phase pattern
        phases_grayscale = self.slm.phases.read() * 256 / (2*np.pi)
        filepath = f"{self.output_filepath}_{self.counter:05d}.png"
        imsave(filepath, phases_grayscale, cmap=self.cmap, vmin=0, vmax=255)
        self.counter += 1

        # Pass on data
        return source_data


class Squarer(Processor):
    """Processor that squares the input."""
    def __init__(self, source):
        super().__init__(source, multi_threaded=False)
        self.source = source

    def _fetch(self, source_data: np.ndarray) -> np.ndarray:
        return source_data**2
