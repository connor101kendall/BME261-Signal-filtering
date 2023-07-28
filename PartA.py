import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

raw_data = pd.read_csv('EMG_Datasets.csv')
data = np.array(raw_data) 
time = data[:, 0]
volt_relaxed = data[:, 1]
volt_contracted = data[:, 2]

# bandstop filter
order = 8
fs = 2000  # Sampling frequency 
f1 = 56  # Lower cutoff 
f2 = 63  # Upper cutoff 
sos = signal.butter(order, [f1, f2], btype='bandstop', fs=fs, output='sos')
volt_relaxed_filtered = signal.sosfilt(sos, volt_relaxed)
volt_contracted_filtered = signal.sosfilt(sos, volt_contracted)

# Calculate the FFT of the filtered relaxed and contracted data
fft_relaxed_filtered = fft(volt_relaxed_filtered)
fft_contracted_filtered = fft(volt_contracted_filtered)
# Calculate the FFT of the filtered relaxed and contracted data
fft_relaxed = fft(volt_relaxed)
fft_contracted = fft(volt_contracted)
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
plt.plot(time, volt_contracted, label='Contracted (Original)')
plt.plot(time, volt_relaxed, label='Relaxed (Original)')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.legend()

plt.figure()
plt.plot(time, volt_contracted_filtered, label='Contracted (Filtered)')
plt.plot(time, volt_relaxed_filtered, label='Relaxed (Filtered)')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.legend()

plt.show()

# Calculate the root mean square (RMS) for filtered relaxed and contracted data separately
rms_relaxed = np.sqrt(np.mean(volt_relaxed**2))
rms_contracted = np.sqrt(np.mean(volt_contracted**2))
rms_relaxed_filtered = np.sqrt(np.mean(volt_relaxed_filtered**2))
rms_contracted_filtered = np.sqrt(np.mean(volt_contracted_filtered**2))

print(f"Root Mean Square (RMS) for relaxed data: {rms_relaxed}")
print(f"Root Mean Square (RMS) for contracted data: {rms_contracted}")
print(f"Root Mean Square (RMS) for filtered relaxed data: {rms_relaxed_filtered}")
print(f"Root Mean Square (RMS) for filtered contracted data: {rms_contracted_filtered}")