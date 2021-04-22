from os import listdir
import numpy as np
import mne

COLUMNS = ['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'CM', 'A1', 'Fp1',
           'Fp2', 'T3', 'T5', 'O1', 'O2', 'X3', 'X2', 'F7', 'F8', 'X1', 'A2',
           'T6', 'T4']
HERTZ = 300


def preprocess_file(filename, input_directory, output_directory):
    data = np.loadtxt(input_directory + '/' + filename, delimiter=',',
                      skiprows=1, usecols=[*range(0, 25)])
    times = data[:, 0]

    transposed_sensors = np.transpose(data[:, 1:])

    # Rescale baseline
    rescaled = mne.baseline.rescale(transposed_sensors, times, (0, .1))

    info = mne.create_info(COLUMNS, HERTZ, ch_types='eeg')
    raw = mne.io.RawArray(rescaled, info)

    # Remove unused sensors
    raw.drop_channels(['X1', 'X2', 'X3'])

    # Lowpass 50Hz, Highpass 0.5Hz
    raw.filter(0.5, 50., fir_design='firwin')

    # Rereference to common mode follower
    raw.set_eeg_reference(ref_channels=['CM'])

    # Remove common channel after using it as a reference
    raw.drop_channels(['CM'])

    name = output_directory + '/' + 'preprocessed_' + filename
    header = str(raw.ch_names).strip("[]").replace("'", "")

    np.savetxt(name, np.transpose(raw.get_data()), delimiter=",",
               header=header, comments="")


def preprocess_files_in_directory(input_directory, output_directory):
    filenames = listdir(input_directory)
    for filename in filenames:
        preprocess_file(filename, input_directory, output_directory)


preprocess_files_in_directory('_data_active_unprocessed',
                              '_data_active_preprocessed')
preprocess_files_in_directory('_data_passive_unprocessed',
                              '_data_passive_preprocessed')