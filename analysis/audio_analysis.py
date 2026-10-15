import os
import librosa
import numpy as np

DATASET_PATH = "dataset/Baby Dataset"

audio_files = []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
            audio_files.append(os.path.join(root, file))

durations = []
sample_rates = []

for file_path in audio_files:
    try:
        y, sr = librosa.load(file_path, sr=None)

        duration = librosa.get_duration(y=y, sr=sr)

        durations.append(duration)
        sample_rates.append(sr)

    except Exception as e:
        print("Error:", file_path)
        print(e)

print("\n========== AUDIO ANALYSIS ==========")

print("\nTOTAL FILES:", len(audio_files))

print("\nSAMPLE RATES:")
for sr in sorted(set(sample_rates)):
    print(sr, "Hz")

print("\nDURATION:")
print("Minimum:", round(min(durations), 2), "seconds")
print("Maximum:", round(max(durations), 2), "seconds")
print("Average:", round(np.mean(durations), 2), "seconds")

print("\nMEDIAN:", round(np.median(durations), 2), "seconds")