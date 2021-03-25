
from scipy.fft import fft, fftfreq
from fourier import partition_eeg_bands
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


EEG_BANDS = {'Delta': (0, 4),
             'Theta': (4, 8),
             'Alpha': (8, 12),
             'Beta': (12, 30),
             'Gamma': (30, 45)}

filedata = pd.read_csv('_data/subject1_eyesclosed.csv')
filedata.drop(columns=['Trigger',
                       'Time_Offset',
                       'ADC_Status',
                       'ADC_Sequence',
                       'Event',
                       'Comments'],
              inplace=True)

data = (filedata[filedata.Time < 2.0]['P3']).to_numpy()
partition_eeg_bands(data, 0.0033, True)

# Number of sample points
N = 599  # 606
# sample spacing
T = 1.0 / (599/2)  # 303.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
temp = np.sin(16.0*2.0*np.pi*x)
temp2 = np.random.uniform(0, 100, N)
temp3 = (np.sin(2.0*2.0*np.pi*x)*100+3500 +
         np.sin(6.0*2.0*np.pi*x)*25+3500 +
         np.sin(10.0*2.0*np.pi*x)*50+3500 +
         np.sin(21.0*2.0*np.pi*x)*25+3500 +
         np.sin(37.5*2.0*np.pi*x)*100+3500)

y = data

# Normalize volume
reduceby = np.min(y) + (np.max(y)-np.min(y))/2
y = np.subtract(y, reduceby)

plt.plot(y)
plt.show()
yf = fft(y)
xf = fftfreq(N, T)[:N//2]
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))

eeg_band_fft = {}
for band in EEG_BANDS:
    frequency_index = np.where((xf >= EEG_BANDS[band][0]) &
                               (xf <= EEG_BANDS[band][1]))[0]
    eeg_band_fft[band] = np.sum(np.abs(yf[frequency_index]))

df = pd.DataFrame(columns=['band', 'val'])
df['band'] = EEG_BANDS.keys()
df['val'] = [eeg_band_fft[band] for band in EEG_BANDS]

ax = df.plot.bar(x='band', y='val', legend=False)
