"""
Batch compare WFS algorithms.

Do multiple wavefront shaping measurements at different locations and compare different algorithms. The motorized stage
is used to aim at different locations. A 'small' FOV laser scanner object provides feedback for the WFS algorithms.
After optimization, a 'large' FOV laser scanner object is used to get quality images.

Note: When newly running this script, make sure the defined file and folder paths are still valid, and update if required.
"""
# Built-in
import time
from pathlib import Path

# External (3rd party)
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import h5py
from tqdm import tqdm
from nidaqmx.constants import TerminalConfiguration
from zaber_motion import Units
from zaber_motion.ascii import Connection

# External (ours)
from openwfs.processors import SingleRoi
from openwfs.devices import ScanningMicroscope, Gain, SLM, Axis
from openwfs.utilities import Transform
from openwfs.algorithms import CustomBlindDualReference

# Internal
from tpmpy.filters import DigitalNotchFilter
from tpmpy.helper_classes import RandomSLMShutter, OffsetRemover
from tpmpy.helper_functions import autodelay_scanner, gitinfo, get_com_by_vid_pid, converge_parking_spot, park_beam, \
    measure_contrast_enhancement, get_dict_from_hdf5


# ====== Settings ====== #
# Saving
save_path = Path('C:/LocalData/')
filename_prefix = 'wfs-comparison_'

# Import modes
print('\nStart import modes...')
modes_filepath = '//ad.utwente.nl/TNW/BMPI/Data/Daniel Cox/ExperimentalData/wfs-OrthoFBDR-comparison/ortho-plane-waves-hires.hdf5'

# Import variables
with h5py.File(modes_filepath, 'r') as f:
    phases_pw_half = f['init_phases_hr'][:,:,:,0,0]
    phases_ortho_pw_half = f['new_phases_hr'][:,:,:,0,0]
    git_info_process_modes = f['git_info']
    git_info_orthonormalization = f['git_info_orthonormalization']

mask_shape = phases_pw_half.shape[0:2]

phases_pw = np.concatenate((phases_pw_half, np.zeros(shape=phases_pw_half.shape)), axis=1)
phases_ortho_pw = np.concatenate((phases_ortho_pw_half, np.zeros(shape=phases_pw_half.shape)), axis=1)
split_mask = np.concatenate((np.zeros(shape=mask_shape), np.ones(shape=mask_shape)), axis=1)


# WFS settings
algorithms = [CustomBlindDualReference, CustomBlindDualReference]
algorithm_kwargs = [{'phases': (phases_pw, np.flip(phases_pw))},
                    {'phases': (phases_ortho_pw, np.flip(phases_ortho_pw))}]
algorithm_common_kwargs = {'iterations': 6, 'phase_steps': 16, 'set1_mask': split_mask, 'do_try_full_patterns': True,
                           'progress_bar_kwargs': {'ncols': 60, 'leave': False}}

np.savez(
    save_path.joinpath(f'{filename_prefix}algorithm_kwargs_t{round(time.time())}'),
    algorithm_kwargs=[algorithm_kwargs],
)

# Stage
stage_settings = {
    'settle_time': 2 * 60 * u.s,
    'step_size': 250 * u.um,
    'num_steps_axis1': 5,
    'num_steps_axis2': 5,
}

# # Uncomment for quick test
# stage_settings = {
#     'settle_time': 2 * u.s,
#     'step_size': 250 * u.um,
#     'num_steps_axis1': 2,
#     'num_steps_axis2': 1,
# }
# algorithm_kwargs = [{'phases': (phases_pw[:,:,0:3], np.flip(phases_pw[:,:,0:3]))},
#                     {'phases': (phases_ortho_pw[:,:,0:3], np.flip(phases_ortho_pw[:,:,0:3]))}]
# algorithm_common_kwargs = {'iterations': 2, 'phase_steps': 4, 'set1_mask': split_mask, 'do_try_full_patterns': True,
#                            'progress_bar_kwargs': {'ncols': 60, 'leave': False}}

# PMT Amplifier
signal_gain = 0.6 * u.V

# SLM
slm_props = {
    'wavelength': 804 * u.nm,
    'f_obj': 12.5 * u.mm,
    'NA_obj': 0.8,
    'slm_to_pupil_magnification': 2,
    'pixel_pitch': 9.2 * u.um,
    'height_pix': 1152,
}

# SLM computed properties
slm_props['height'] = slm_props['pixel_pitch'] * slm_props['height_pix']
slm_props['pupil_radius_on_slm'] = slm_props['f_obj'] * slm_props['NA_obj'] / slm_props['slm_to_pupil_magnification']
slm_props['scaling_factor'] = 2 * slm_props['pupil_radius_on_slm'] / slm_props['height']

# Notch filter
notch_kwargs = {
    'frequency': 320 * u.kHz,
    'bandwidth': 15 * u.kHz,
    'dtype_out': 'int16',
}

# Laser scanner
scanner_props = {
    'sample_rate': 0.8 * u.MHz,
    'resolution': 1024,
    'zoom': 15,
    'initial_delay': 100.0 * u.us,
    'scale': 440 * u.um / u.V,
    'v_min': -1.0 * u.V,
    'v_max': 1.0 * u.V,
    'maximum_acceleration': 5.0e4 * u.V/u.s**2,
}

park_kwargs = {
    'do_plot': False,                # For debugging
    'median_filter_size': (3, 3),
    'target_width': 64,
    'max_iterations': 15,
    'park_to_one_pixel': False,
}

roi_kwargs = {
    'radius': 31,
    'pos': (32, 32),
}


# ====== Prepare hardware ====== #
print('Start hardware initialization...')

x_axis = Axis(channel='Dev4/ao3',
              v_min=scanner_props['v_min'],
              v_max=scanner_props['v_max'],
              terminal_configuration=TerminalConfiguration.DEFAULT,
              maximum_acceleration=scanner_props['maximum_acceleration'])

y_axis = Axis(channel='Dev4/ao2',
              v_min=scanner_props['v_min'],
              v_max=scanner_props['v_max'],
              terminal_configuration=TerminalConfiguration.DEFAULT,
              maximum_acceleration=scanner_props['maximum_acceleration'])

# Define laser scanner, with offset
scanner_with_offset = ScanningMicroscope(
    bidirectional=True,
    sample_rate=scanner_props['sample_rate'],
    y_axis=y_axis,
    x_axis=x_axis,
    input=('Dev4/ai16', -1.0 * u.V, 1.0 * u.V),
    scale=scanner_props['scale'],
    delay=scanner_props['initial_delay'],
    preprocessor=DigitalNotchFilter(**notch_kwargs),
    reference_zoom=2.0,
    resolution=scanner_props['resolution'])  # Define notch filter to remove background ripple
scanner_with_offset.zoom = scanner_props['zoom']

# Define Processor that fetches data from scanner and removes offset and ROI detector
reader = OffsetRemover(source=scanner_with_offset, offset=2 ** 15, dtype_out='float64')
roi = SingleRoi(reader, **roi_kwargs)

# SLM
slm_shape = (slm_props['height_pix'], slm_props['height_pix'])
slm_transform = Transform(((slm_props['scaling_factor'], 0.0), (0.0, slm_props['scaling_factor'])))
slm = SLM(2, transform=slm_transform)
slm.lookup_table = np.arange(0, 0.2623 * slm_props['wavelength'].to_value(u.nm) - 23.33)  # Temporary, hardcoded LUT
shutter = RandomSLMShutter(slm)
print('Found SLM')

# Define NI-DAQ Gain channel and settings
gain_amp = Gain(
    port_ao="Dev4/ao0",
    port_ai="Dev4/ai0",
    port_do="Dev4/port0/line0",
    reset=False,
    gain=0.00 * u.V,
)
gain_amp.gain = signal_gain
print('Connected to Gain NI-DAQ')


# ====== Preparation measurements ====== #
shutter.open = False
input(f'Please unblock laser and press enter')

# Determine and update scanner delays
print('Determining bidirectional delay of scanner...')
scanner_props['delay'] = autodelay_scanner(shutter=shutter, image_reader=reader, scanner=reader.source)
print(f"Scanner delay: {scanner_props['delay']}")

# Zaber stage
comport = get_com_by_vid_pid(vid=0x2939, pid=0x495b)                # Get COM-port of Zaber X-MCB2

with Connection.open_serial_port(comport) as connection:            # Open connection with Zaber stage
    # Zaber stage initialization
    connection.enable_alerts()
    device_list = connection.detect_devices()
    device = device_list[0]
    print(f"Connected to {device.name} (serial number: {device.serial_number}) at {comport}")
    axis1 = device.get_axis(1)
    axis2 = device.get_axis(2)
    axis1_start_position = axis1.get_position()
    axis2_start_position = axis2.get_position()

    # Repeat experiment on different locations. Move with Zaber stage.
    total_steps = stage_settings['num_steps_axis1'] * stage_settings['num_steps_axis2']
    progress_bar = tqdm(colour='blue', total=total_steps, ncols=60)
    for a1 in range(stage_settings['num_steps_axis1']):             # Loop over stage axis 1
        for a2 in range(stage_settings['num_steps_axis2']):         # Loop over stage axis 2
            print(f'\nStart measurement at axes pos. {a1}/{stage_settings["num_steps_axis1"]}, '
                    + f'{a2}/{stage_settings["num_steps_axis2"]}')

            print('Start converging to parking spot')
            park_location, park_imgs = converge_parking_spot(shutter=shutter, image_reader=reader,
                                                             scanner=reader.source, **park_kwargs)
            print(f'Beam parking spot at {park_location}')

            wfs_results_all = [None] * len(algorithms)
            contrast_results_all = [None] * len(algorithms)
            signal_flat = [None] * len(algorithms)
            signal_shaped = [None] * len(algorithms)

            for n_alg, alg_constructor in enumerate(algorithms):     # Loop over algorithms
                # Construct algorithm
                alg = alg_constructor(feedback=roi, slm=slm, slm_shape=slm_shape, **algorithm_common_kwargs,
                                      **algorithm_kwargs[n_alg])

                print(f'Start Alg.{n_alg} - {alg_constructor.__name__}...')
                park_beam(scanner_with_offset, park_location)

                # Flat wavefront signal
                shutter.open = True
                slm.set_phases(0)
                signal_flat[n_alg] = reader.read()

                # Run WFS measurement
                wfs_result = alg.execute()

                # Shaped wavefront signal
                shaped_phases = -np.angle(wfs_result.t)
                slm.set_phases(shaped_phases)
                signal_shaped[n_alg] = reader.read()
                shutter.open = False

                # Report and save
                print(f'\nSignal enhancement: {signal_shaped[n_alg].mean() / signal_flat[n_alg].mean():.3f}')
                wfs_results_all[n_alg] = wfs_result

                # Full frame measurement
                print('Measure contrast enhancement...')
                scanner_with_offset.reset_roi()
                contrast_results = measure_contrast_enhancement(shutter, reader, scanner_with_offset, slm, shaped_phases)
                print(f'Contrast enhancement: {contrast_results["contrast_enhancement"]:.3f}')
                contrast_results_all[n_alg] = contrast_results

            # Save results
            # TODO: switch to HDF5 and/or json
            print('Save...')
            park_result = {
                'location': park_location,
                'imgs': park_imgs,
            }
            np.savez(
                save_path.joinpath(f'{filename_prefix}t{round(time.time())}'),
                wfs_results_all=[wfs_results_all],
                contrast_results_all=[contrast_results_all],
                gitinfo=[gitinfo()],
                gitinfo_process_modes=[git_info_process_modes],
                gitinfo_orthonormalization=[git_info_orthonormalization],
                time=time.time(),
                modes_filepath=modes_filepath,
                algorithm_types=[a.__name__ for a in algorithms],
                algorithm_common_kwargs=[algorithm_common_kwargs],
                stage_settings=[stage_settings],
                signal_gain=[{'gain': signal_gain}],
                slm_props=[slm_props],
                notch_kwargs=[notch_kwargs],
                scanner_props=[scanner_props],
                park_kwargs=[park_kwargs],
                park_result=[park_result],
                roi_kwargs=[roi_kwargs],
                signal_flat=[signal_flat],
                signal_shaped=[signal_shaped],
            )

            progress_bar.update()

            print('\nMove stage')

            if a2+1 < stage_settings['num_steps_axis2']:
                # Move stage axis and let it settle
                axis2.move_relative(stage_settings['step_size'].to_value(u.um), Units.LENGTH_MICROMETRES)
                time.sleep(stage_settings['settle_time'].to_value(u.s))
            else:
                break                                   # Skip last stage move and sleep

        axis2.move_absolute(axis2_start_position)       # Return stage axis to starting position
        if a1+1 < stage_settings['num_steps_axis1']:
            # Move stage axis and let it settle
            axis1.move_relative(stage_settings['step_size'].to_value(u.um), Units.LENGTH_MICROMETRES)
            time.sleep(stage_settings['settle_time'].to_value(u.s))
        else:
            break                                       # Skip last sleep and stage movement

    axis1.move_absolute(axis1_start_position)           # Return stage axis to starting position


print('--- Done! ---')
input('Please block laser and press enter')
shutter.open = True

pass
