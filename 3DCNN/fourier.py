from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


EEG_BANDS = {'Delta': (0, 4),
             'Theta': (4, 8),
             'Alpha': (8, 12),
             'Beta': (12, 30),
             'Gamma': (30, 45)}


def sum_band_data(ft_data, ft_x):
    eeg_band_fft = {}
    for band in EEG_BANDS:
        frequency_index = np.where((ft_x >= EEG_BANDS[band][0]) &
                                   (ft_x <= EEG_BANDS[band][1]))[0]
        eeg_band_fft[band] = np.sum(np.abs(ft_data[frequency_index]))

    return eeg_band_fft


def partition_eeg_bands(data, sample_rate, plot=False):
    num_samples = len(data)

    # Normalize volume about x axis
    minimum = np.min(data)
    subtrahend = minimum + (np.max(data)-minimum) / 2
    data = np.subtract(data, subtrahend)

    if plot:
        plt.figure(1)
        plt.title('Raw EEG data')
        plt.plot(data)

    ft_data = fft(data)
    ft_x = fftfreq(num_samples, sample_rate)[:num_samples//2]

    if plot:
        plt.figure(2)
        plt.title('Fourier transformed EEG data')
        plt.plot(ft_x,
                 2.0/num_samples * np.abs(ft_data[0:num_samples//2]))

    eeg_band_fft = sum_band_data(ft_data, ft_x)

    if plot:
        df = pd.DataFrame(columns=['band', 'value'])
        df['band'] = EEG_BANDS.keys()
        df['value'] = [eeg_band_fft[band] for band in EEG_BANDS]

        df.plot.bar(x='band', y='value', legend=False,
                    title='EEG wave distribution')
        plt.show(block=True)

    return eeg_band_fft


def test_data(filepath='_data/subject1_eyesclosed.csv',
              column='P3',
              time_start=0.0,
              time_end=2.0):
    filedata = pd.read_csv(filepath)
    filedata.drop(columns=['Trigger',
                           'Time_Offset',
                           'ADC_Status',
                           'ADC_Sequence',
                           'Event',
                           'Comments'],
                  inplace=True)

    data = (filedata[filedata['Time']
            .between(time_start, time_end)][column]
            .to_numpy())
    partition_eeg_bands(data, 0.0033, True)


def test_data_stew(filepath='_data_stew/sub01_hi.txt',
                   column=0,
                   time_start=0.0,
                   time_end=2.0):
    filedata = pd.read_csv(filepath, delim_whitespace=True,
                           index_col=None, header=None)
    hz = 128
    data = (filedata[int(time_start * hz):int(time_end * hz)][column]
            .to_numpy())
    partition_eeg_bands(data, 1 / hz, True)
