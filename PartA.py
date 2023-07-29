import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

raw_data = pd.read_csv('EMG_Datasets.csv')
data = np.array(raw_data) 
time = data[:, 0]
relaxed = data[:, 1]
contracted = data[:, 2]

# bandstop filter
order = 4
fs = 2000  # Sampling frequency 
f1 = 55  # Lower cutoff 
f2 = 63  # Upper cutoff 
sos = signal.butter(order, [f1, f2], btype='bandstop', fs=fs, output='sos')
relaxed_filtered = signal.sosfilt(sos, relaxed)
contracted_filtered = signal.sosfilt(sos, contracted)

# Calculate the FFT of the filtered relaxed and contracted data
fft_relaxed_filtered = fft(relaxed_filtered)
fft_contracted_filtered = fft(contracted_filtered)
# Calculate the FFT of the filtered relaxed and contracted data
fft_relaxed = fft(relaxed)
fft_contracted = fft(contracted)
# Calculate the corresponding frequencies for the FFT
freq = fftfreq(len(time), 1.0 / fs)

# Plot the amplitude spectra of the filtered data
plt.figure()
plt.plot(freq, np.abs(fft_relaxed_filtered), label='Relaxed (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of Filtered Relaxed Data')
plt.xlim([0,400])
plt.legend()

plt.figure()
plt.plot(freq, np.abs(fft_contracted_filtered), label='Contracted (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of Filtered Contracted Data')
plt.xlim([0,400])
plt.legend()

plt.figure()
plt.plot(freq, np.abs(fft_relaxed), label='Relaxed')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of Relaxed Data')
plt.xlim([0,400])
plt.legend()

plt.figure()
plt.plot(freq, np.abs(fft_contracted), label='Contracted')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of Contracted Data')
plt.xlim([0,400])
plt.legend()



# Plot the original and filtered data
plt.figure()
plt.plot(time, contracted, label='Contracted (Original)')
plt.plot(time, relaxed, label='Relaxed (Original)')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.legend()

plt.figure()
plt.plot(time, contracted_filtered, label='Contracted (Filtered)')
plt.plot(time, relaxed_filtered, label='Relaxed (Filtered)')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.legend()

plt.show()

# Calculate the root mean square (RMS) for filtered relaxed and contracted data separately
rms_relaxed = np.sqrt(np.mean(relaxed**2))
rms_contracted = np.sqrt(np.mean(contracted**2))
rms_relaxed_filtered = np.sqrt(np.mean(relaxed_filtered**2))
rms_contracted_filtered = np.sqrt(np.mean(contracted_filtered**2))

print(f"Root Mean Square (RMS) for relaxed data: {rms_relaxed}")
print(f"Root Mean Square (RMS) for contracted data: {rms_contracted}")
print(f"Root Mean Square (RMS) for filtered relaxed data: {rms_relaxed_filtered}")
print(f"Root Mean Square (RMS) for filtered contracted data: {rms_contracted_filtered}")