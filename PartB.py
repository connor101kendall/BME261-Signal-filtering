import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import scipy.io.wavfile as wavfile

# Load the .wav 
sampling_rate, audio_data = wavfile.read('quote.wav')

# If the sampling rate is > 16KHz, down-sample to 16kHz
target_sampling_rate = 16000  # Desired sampling rate
if sampling_rate > target_sampling_rate:
    downsample_factor = sampling_rate // target_sampling_rate
    audio_data = audio_data[::downsample_factor]
    sampling_rate = target_sampling_rate
 
# If it is a stereo recording convert to mono
if len(audio_data.shape) > 1 and audio_data.shape[1] == 2:
    audio_data = audio_data[:, 0]

# Time Segmentation
chunk_duration_ms = 20  # Duration of each chunk in milliseconds

# Convert chunk_duration_ms to the number of samples based on the sampling rate
chunk_duration_samples = int(sampling_rate * (chunk_duration_ms / 1000))

# Create an array of time indices for each chunk
time_indices = np.arange(0, len(audio_data), chunk_duration_samples)

# Divide the audio data into chunks
audio_chunks = []
for i in range(len(time_indices) - 1):
    chunk = audio_data[time_indices[i]:time_indices[i + 1]]
    audio_chunks.append(chunk)

# Plot the original .wav file
plt.figure()
time = np.arange(0, len(audio_data)) / sampling_rate
plt.plot(time, audio_data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Audio')
 #plt.show()

# Define parameters for the band-pass filters
num_filters = 150  # Number of band-pass filters
center_frequencies = np.linspace(100, 5000, num_filters)  # place equally spaced center frequencies from 100 Hz to 5 kHz
bandwidth = 50  # Bandwidth 
filter_order = 4  # Order

# Create an array to store the RMS values of each band-pass filter's output for each chunk
rms_values = np.zeros((len(audio_chunks), num_filters))

# Loop over each chunk and apply band-pass filters
for i, chunk in enumerate(audio_chunks):
    # Apply a bank/array of band-pass filters (BPFs)
    sos_bandpass_filters = []
    for freq in center_frequencies:
        f1 = freq - (bandwidth / 2)
        f2 = freq + (bandwidth / 2)
        sos_bandpass = signal.butter(filter_order, [f1, f2], btype='bandpass', fs=sampling_rate, output='sos')
        sos_bandpass_filters.append(sos_bandpass)

    # Array to store the filtered output of each BPF
    bpf_outputs = np.zeros((num_filters, len(chunk)))

    # Apply each band-pass filter to the chunk
    for j, sos in enumerate(sos_bandpass_filters):
        bpf_outputs[j] = signal.sosfilt(sos, chunk)

    # Calculate the RMS of the output of each BPF for the current chunk
    rms_values[i] = np.sqrt(np.mean(bpf_outputs**2, axis=1))

# Plot the RMS values of each BPF for each chunk
plt.figure(figsize=(12, 6))
for i in range(num_filters):
    plt.plot(np.arange(len(rms_values)), rms_values[:, i], label=f'Filter {i+1}')
plt.xlabel('Chunk Index')
plt.ylabel('RMS')
plt.title('RMS Values of Band-Pass Filter Outputs for Each Chunk')
plt.legend()
#plt.show()

# Synthesize the sine waves and superimpose them for each chunk
synthesized_chunks = []
for i, chunk in enumerate(audio_chunks):
    # Synthesize the sine-waves
    synthesized_wave = np.zeros_like(chunk, dtype=float)
    for j, sos in enumerate(sos_bandpass_filters):
        rms = rms_values[i, j]
        frequency = center_frequencies[j]
        time = np.arange(0, len(chunk)) / sampling_rate
        synthesized_wave += rms * np.sin(2 * np.pi * frequency * time)

    # Add up the synthesized waves from all the bands
    synthesized_chunks.append(synthesized_wave)

# Combine the synthesized chunks into a single time stream
synthesized_audio = np.concatenate(synthesized_chunks)

# Save the synthesized audio as a .wav file
output_filename = 'robo.wav'
wavfile.write(output_filename, sampling_rate, synthesized_audio.astype(np.int16))

print("Synthesized audio saved as:", output_filename)