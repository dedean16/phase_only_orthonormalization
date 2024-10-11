from typing import Tuple, Dict

import numpy as np
from numpy import ndarray as nd
from numpy.fft import fft, ifft
import scipy
import matplotlib.pyplot as plt
from serial.tools import list_ports

from openwfs import Processor
from openwfs.devices import ScanningMicroscope, SLM
from openwfs.algorithms.troubleshoot import contrast_enhancement


def get_com_by_vid_pid(vid: int, pid: int):
    """
    Get COM port by USB vid and pid.

    Args:
        vid: USB vendor ID.
        pid: USB product ID.

    Returns:
        The COM port as a string, e.g. "COM3".

    Find USB vendor and product IDs (hexadecimal):
        https://devicehunt.com/all-usb-vendors
        https://the-sz.com/products/usbid/index.php
        http://www.linux-usb.org/usb.ids
    """
    for port in list_ports.comports():
        if port.vid == vid and port.pid == pid:
            return port.name
    return None


def find_scan_delay(img: np.ndarray, axis_cross_corr: int = 1, axis_sum: int = 0):
    """
    Find the scan delay in pixels from a bidirectionally scanned image, to align the even and odd lines.

    Args:
        img: Image to find scan delay of.
        axis_cross_corr: Array axis to find delay of.
        axis_sum: Array axis to take the sum over.

    Returns:
        The required delay in pixels.
    """
    even = img[0::2]
    odd = img[1::2]
    assert even.shape == odd.shape

    # Image pixel shift between even and odd lines
    cross_corr = ifft(fft(odd, axis=axis_cross_corr).conj() * fft(even, axis=axis_cross_corr), axis=axis_cross_corr)
    index_max = np.argmax(np.abs(cross_corr).sum(axis=axis_sum))
    s = even.shape[axis_cross_corr]
    pixel_shift = index_max if index_max < s / 2 else index_max - s
    delay_pix = pixel_shift / 2  # Both lines are shifted -> only half the shift is required
    return delay_pix


def autodelay_scanner(shutter, image_reader: Processor, scanner: ScanningMicroscope) -> float:
    """
    Find and set the delay for given scanner.

    Args:
        shutter: Shutter object with an open property.
        image_reader: Scanner object with a .read() method to read the image.
        scanner: The scanner whose ROI must be adjusted.

    Returns:
        The delay that was set in pixels.
    """
    shutter.open = True
    img_interlaced = image_reader.read()
    shutter.open = False
    relative_delay_pix = find_scan_delay(img_interlaced)
    delay_pix = scanner.delay + relative_delay_pix * scanner.dwell_time
    scanner.delay = delay_pix
    return delay_pix


def park_beam(scanner, park_spot: Tuple[int, int, int, int]):
    """Park the beam at pixel location (left, top, width, height)."""
    scanner.left = park_spot[0]
    scanner.top = park_spot[1]
    scanner.width = park_spot[2]
    scanner.height = park_spot[3]
    if park_spot[3] == 1:
        scanner._bidirectional = False


def find_parking_spot(shutter: Processor, image_reader, median_filter_size: Tuple[int, int],
                      edge_skip: int) -> Tuple:
    """
    Find the maximum value (after some filtering) in the image and find a parking spot for the beam.
    Note: The object of interest should be at least the size of median_filter_size.

    Args:
        shutter: Shutter object
        image_reader: Scanner object with a .read() method to read the image.
        median_filter_size: Size of the median filter in (pixels, pixels)
        edge_skip: Number of pixels from the edge to skip

    Returns:
        location: Found parking spot
        img_filt: Filtered image used to find parking spot
    """
    # Get image
    shutter.open = True
    img = image_reader.read()
    shutter.open = False

    # Median filter
    img_filt = scipy.ndimage.median_filter(img, size=median_filter_size)

    # Skip edges
    min_value = np.min(img)
    img_filt[:edge_skip, :] = min_value
    img_filt[-edge_skip:, :] = min_value
    img_filt[:, :edge_skip] = min_value
    img_filt[:, -edge_skip:] = min_value

    # Get indices of maximum
    s = img_filt.shape  # Get shape
    max_location = np.unravel_index(np.argmax(img_filt), s)  # Find 2D indices of maximum

    return max_location, img_filt


def converge_parking_spot(shutter, image_reader: Processor, scanner: ScanningMicroscope,
                          median_filter_size: Tuple[int, int], target_width: int, max_iterations: int,
                          do_plot: bool = True, park_to_one_pixel: bool = True) \
        -> Tuple[Tuple[int, int, int, int], nd]:
    """
    Converge to a parking spot by iteratively halving the ROI and centering the new ROI around the maximum. Useful when
    the laser scanner galvo response is non-linear. Padding is also halved every iteration.
    Note: Skips edges of (image width / 4) to prevent setting out of bounds ROI.

    Args:
        shutter: Shutter object with an open property.
        image_reader: Processor with a .read() method to read the image. May be a ScanningMicroscope or a
            post-processing Processor with the ScanningMicroscope as (sub)source.
        scanner: The scanner whose ROI must be adjusted.
        median_filter_size: Size of the median filter in (pixels, pixels)
        target_width: Width to converge to.
        max_iterations: Maximum number of iterations. Must be at least 1.
        do_plot: Plot frames while converging to a parking spot. For debugging.
        park_to_one_pixel: After target_width is reached, drop to width=height=1 for true beam parking.

    Returns:
        park_spot: Found parking spot (left, top, width, height)
        imgs: All filtered images used to find the parking spot
    """

    imgs = []

    if do_plot:
        plt.figure()

    for it in range(max_iterations):
        autodelay_scanner(shutter, image_reader, scanner)

        # Find parking spot
        edge_skip = scanner.width // 4
        max_location, img = find_parking_spot(shutter, image_reader, median_filter_size, edge_skip)
        imgs += [img]

        if do_plot:
            plt.clf()
            vmax = img.max() * 0.9
            plt.imshow(img, vmin=0, vmax=vmax)
            plt.plot(max_location[1], max_location[0], '+r')
            plt.title(f'Mean signal: {img.max():.1f}')
            plt.pause(1.0)

        # Halve ROI size
        scanner.width /= 2
        scanner.height /= 2

        # Center ROI around newly found parking spot
        left = scanner.left + max_location[1] - scanner.width // 2
        top = scanner.top + max_location[0] - scanner.height // 2
        scanner.left = np.maximum(0, left)
        scanner.top = np.maximum(0, top)

        if scanner.width <= target_width:
            break

    if park_to_one_pixel:
        scanner.width = 1
        scanner.height = 1
        left = scanner.left + max_location[1]
        top = scanner.top + max_location[0]
        scanner.left = np.maximum(0, left)
        scanner.top = np.maximum(0, top)

    return (scanner.left, scanner.top, scanner.width, scanner.height), imgs


def measure_contrast_enhancement(shutter, reader: Processor, scanner: ScanningMicroscope, slm: SLM, phase_pattern: nd) \
        -> Dict:
    """
    Measure contrast enhancement.

    Args:
        shutter: Object that controls the shutter by setting the open property.
        scanner: Object that controls the laser scanner.
        reader: Processor that reads the laser scanner. May be the laser scanner itself or a post processing Processor.
        slm: Spatial light modulator object.
        phase_pattern: Phases to display on the SLM.

    Returns:
        img_flat_wf: Image with flat wavefront.
        img_shaped_wf: Image with shaped wavefront.
        contrast_enhancement: Unbiased contrast enhancement with given phase pattern.
    """
    # Noise measurement
    shutter.open = False
    noise = reader.read()

    # Flat wavefront
    shutter.open = True
    slm.set_phases(0)
    img_flat_wf = reader.read()

    # Shaped wavefront
    slm.set_phases(phase_pattern)
    img_shaped_wf = reader.read()
    shutter.open = False

    contrast_results = {
        'img_flat_wf': img_flat_wf,
        'img_shaped_wf': img_shaped_wf,
        'contrast_enhancement': contrast_enhancement(img_shaped_wf, img_flat_wf, noise),
    }

    return contrast_results
