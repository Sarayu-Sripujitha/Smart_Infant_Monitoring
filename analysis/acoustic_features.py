import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = "dataset/Baby Dataset"
RESULTS_PATH = "results"

os.makedirs(RESULTS_PATH, exist_ok=True)

audio_file = None

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
            audio_file = os.path.join(root, file)
            break

    if audio_file:
        break

if audio_file is None:
    print("No audio file found.")
    exit()

print("Analyzing:")
print(audio_file)

y, sr = librosa.load(audio_file, sr=16000, mono=True)

print("\n========== AUDIO INFORMATION ==========")
print("Sample Rate:", sr, "Hz")
print("Duration:", round(librosa.get_duration(y=y, sr=sr), 2), "seconds")
print("Number of Samples:", len(y))

# ------------------------------------------------
# 1. WAVEFORM
# ------------------------------------------------

plt.figure(figsize=(12, 4))

librosa.display.waveshow(y, sr=sr)

plt.title("Infant Cry Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()

plt.savefig(os.path.join(RESULTS_PATH, "waveform.png"))
plt.close()

# ------------------------------------------------
# 2. MEL-SPECTROGRAM
# ------------------------------------------------

mel = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_mels=128
)

mel_db = librosa.power_to_db(mel, ref=np.max)

plt.figure(figsize=(12, 5))

librosa.display.specshow(
    mel_db,
    sr=sr,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Infant Cry Mel-Spectrogram")
plt.tight_layout()

plt.savefig(os.path.join(RESULTS_PATH, "mel_spectrogram.png"))
plt.close()

# ------------------------------------------------
# 3. MFCC
# ------------------------------------------------

mfcc = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=13
)

plt.figure(figsize=(12, 5))

librosa.display.specshow(
    mfcc,
    x_axis="time",
    sr=sr
)

plt.colorbar()
plt.title("Infant Cry MFCC")
plt.ylabel("MFCC Coefficient")

plt.tight_layout()

plt.savefig(os.path.join(RESULTS_PATH, "mfcc.png"))
plt.close()

# ------------------------------------------------
# 4. ZERO CROSSING RATE
# ------------------------------------------------

zcr = librosa.feature.zero_crossing_rate(y)

print("\nZero Crossing Rate:")
print("Mean:", round(float(np.mean(zcr)), 5))

# ------------------------------------------------
# 5. SPECTRAL CENTROID
# ------------------------------------------------

centroid = librosa.feature.spectral_centroid(
    y=y,
    sr=sr
)

print("\nSpectral Centroid:")
print("Mean:", round(float(np.mean(centroid)), 2), "Hz")

# ------------------------------------------------
# 6. SPECTRAL BANDWIDTH
# ------------------------------------------------

bandwidth = librosa.feature.spectral_bandwidth(
    y=y,
    sr=sr
)

print("\nSpectral Bandwidth:")
print("Mean:", round(float(np.mean(bandwidth)), 2), "Hz")

# ------------------------------------------------
# 7. SPECTRAL ROLLOFF
# ------------------------------------------------

rolloff = librosa.feature.spectral_rolloff(
    y=y,
    sr=sr
)

print("\nSpectral Rolloff:")
print("Mean:", round(float(np.mean(rolloff)), 2), "Hz")

# ------------------------------------------------
# 8. RMS ENERGY
# ------------------------------------------------

rms = librosa.feature.rms(y=y)

print("\nRMS Energy:")
print("Mean:", round(float(np.mean(rms)), 5))

# ------------------------------------------------
# 9. PITCH / FUNDAMENTAL FREQUENCY
# ------------------------------------------------

f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=80,
    fmax=1000
)

valid_f0 = f0[~np.isnan(f0)]

print("\nPitch / Fundamental Frequency:")

if len(valid_f0) > 0:
    print("Mean:", round(float(np.mean(valid_f0)), 2), "Hz")
    print("Minimum:", round(float(np.min(valid_f0)), 2), "Hz")
    print("Maximum:", round(float(np.max(valid_f0)), 2), "Hz")
else:
    print("Pitch could not be detected.")

# ------------------------------------------------
# MFCC VALUES
# ------------------------------------------------

print("\nMFCC Shape:", mfcc.shape)

print("\nMean MFCC Values:")

mfcc_mean = np.mean(mfcc, axis=1)

for i, value in enumerate(mfcc_mean):
    print(f"MFCC {i + 1}: {value:.4f}")

print("\n=======================================")
print("Acoustic analysis completed.")
print("Graphs saved inside:", RESULTS_PATH)