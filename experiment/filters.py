"""Functions to filter raw data."""
from typing import Optional

import numpy as np
import astropy.units as u
from astropy.units import Quantity


def gaussian(freqs, frequency, bandwidth):
    return np.exp(-(freqs - frequency) ** 2 / (2 * bandwidth ** 2))


class DigitalNotchFilter:
    def __init__(self, frequency: Quantity[u.Hz], bandwidth: Quantity[u.Hz], dtype_out: Optional[str], axis: int = 0):
        """
        Digital notch filter with Gaussian shape. For real signals.

        Args:
            frequency: The center frequency of the notch filter.
            bandwidth: The bandwidth of the notch filter, i.e. the standard deviation of the gaussian window.
            axis: The ndarray axis to apply the filter to.
            dtype_out: The output will be cast to this dtype.

        Returns:
            filtered_data: The filtered data.
        """
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.axis = axis
        self.dtype_out = dtype_out

    def __call__(self, data: np.ndarray, sample_rate: Optional[float]):
        f = np.fft.fft(data, axis=self.axis)
        freqs = np.fft.fftfreq(data.shape[0], d=1/sample_rate)
        gaussian_notch = 1 - gaussian(freqs, self.frequency, self.bandwidth) \
                           - gaussian(freqs, -self.frequency, self.bandwidth)
        filtered_data = np.real(np.fft.ifft(f * gaussian_notch)).astype(dtype=self.dtype_out)
        return filtered_data

    def __repr__(self):
        return f'DigitalNotchFilter:\n  frequency={self.frequency}\n  bandwidth={self.bandwidth}\n  '\
               + f'axis={self.axis}, dtype_out={self.dtype_out}'
