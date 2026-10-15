import os
import librosa

DATASET_PATH = "dataset"

audio_files = []
classes = set()
sample_rates = []
durations = []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):

            file_path = os.path.join(root, file)

            # Class = folder immediately containing the audio file
            label = os.path.basename(root)

            audio_files.append(file_path)
            classes.add(label)

            try:
                y, sr = librosa.load(file_path, sr=None)

                sample_rates.append(sr)
                durations.append(len(y) / sr)

            except Exception as e:
                print("Could not read:", file_path)
                print("Error:", e)

print("\n========== DATASET INFORMATION ==========")

print("\nTOTAL AUDIO FILES:")
print(len(audio_files))

print("\nCLASSES:")
for c in sorted(classes):
    print(c)

print("\nNUMBER OF CLASSES:")
print(len(classes))

print("\nSAMPLE RATES:")
print(sorted(set(sample_rates)))

if durations:
    print("\nDURATION STATISTICS:")
    print("Minimum:", round(min(durations), 2), "seconds")
    print("Maximum:", round(max(durations), 2), "seconds")
    print("Average:", round(sum(durations) / len(durations), 2), "seconds")

print("\n==========================================")