import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import correlate
import sys

# this uses short-time fourier transforms (STFT)

# minimum height used to count a peak as a jump/land, can also be entered as a tuple (a,b) of minimum/maximum threshold respectively
jump_threshold = 0.5
landing_threshold = 0.5

# size (in samples) of segments in STFT 
window_length = 2048

# load reference jump sound
jump_sound_file = 'path to king_jump.wav'
jump_sound, sr_jump = librosa.load(jump_sound_file, sr=None)

print("Sample rate:", sr_jump, flush=True)

# load reference landing sound
landing_sound_file = 'path to king_land.wav'  
landing_sound, sr_landing = librosa.load(landing_sound_file, sr=None)

# load audio from speedrun
speedrun_audio_file = 'path to speedrun audio'  
speedrun_audio, sr_video = librosa.load(speedrun_audio_file, sr=None)

# ensure all audio signals are at the same sample rate
if sr_jump != sr_video:
    print("Resampling speedrun audio to match jump sound sample rate.", flush=True)
    speedrun_audio = librosa.resample(speedrun_audio, orig_sr=sr_video, target_sr=sr_jump)

if sr_landing != sr_video:
    print("Resampling landing sound to match speedrun audio sample rate.", flush=True)
    landing_sound = librosa.resample(landing_sound, orig_sr=sr_landing, target_sr=sr_video)

# compute the STFT for the jump sound, landing sound, and speedrun audio
jump_stft = np.abs(librosa.stft(jump_sound))
landing_stft = np.abs(librosa.stft(landing_sound))
speedrun_stft = np.abs(librosa.stft(speedrun_audio))

# plot STFTs 
plt.figure(figsize=(15, 8))

plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(jump_stft, ref=np.max), sr=sr_jump, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Jump Sound Spectrogram')

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(landing_stft, ref=np.max), sr=sr_landing, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Landing Sound Spectrogram')

plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(speedrun_stft, ref=np.max), sr=sr_video, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Speedrun Audio Spectrogram')

plt.tight_layout()
plt.show(block=False)  

# downsampling STFT arrays for efficiency
jump_stft_avg = np.mean(jump_stft, axis=0)
landing_stft_avg = np.mean(landing_stft, axis=0)
speedrun_stft_avg = np.mean(speedrun_stft, axis=0)

# correlation of jump sound 
correlation_jump = correlate(speedrun_stft_avg, jump_stft_avg)
correlation_jump = correlation_jump / np.max(correlation_jump)  # normalize for better plotting

# plot jump correlation
plt.figure(figsize=(10, 4))
plt.plot(correlation_jump)
plt.title('Cross-Correlation between Jump Sound and Speedrun Audio')
plt.xlabel('Time (frames)')
plt.ylabel('Correlation')
plt.grid(True)

plt.show(block=False)  # show plots without blocking

# correlation of landing sound
correlation_landing = correlate(speedrun_stft_avg, landing_stft_avg)
correlation_landing = correlation_landing / np.max(correlation_landing)  # normalize correlation

# plot landing correlation
plt.figure(figsize=(10, 4))
plt.plot(correlation_landing)
plt.title('Cross-Correlation between Landing Sound and Speedrun Audio')
plt.xlabel('Time (frames)')
plt.ylabel('Correlation')
plt.grid(True)

plt.show(block=False)  # show plots without blocking

reference_jump_length = len(jump_sound)
reference_land_length = len(landing_sound)

min_distance_jump_samples = reference_jump_length  # minimum distance between peaks to avoid double counting, as peaks oscillate
min_distance_jump_frames = min_distance_jump_samples // window_length # convert jump length from samples to frames

min_distance_landing_samples = len(landing_sound)  # length in samples
min_distance_landing_frames = min_distance_landing_samples // window_length  # convert landing length from samples to frames

print("Reference jumping length:", min_distance_jump_frames, flush=True)
print("Reference landing length:", min_distance_landing_frames, flush=True)

print("Jump threshold:", jump_threshold, flush=True)
print("Landing threshold:", landing_threshold, flush=True)

peaks, _ = find_peaks(correlation_jump, height=jump_threshold, distance=min_distance_jump_frames)
landing_peaks, _ = find_peaks(correlation_landing, height=landing_threshold, distance=min_distance_landing_frames)

print(f"Number of jumps detected: {len(peaks)}", flush=True)
print(f"Number of landings detected: {len(landing_peaks)}", flush=True)

plt.show()  
