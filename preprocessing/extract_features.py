import os
import numpy as np
import pandas as pd
import librosa

DATASET_PATH = "dataset/Baby Dataset"
OUTPUT_PATH = "features/features.csv"

TARGET_SR = 16000
N_MFCC = 13

os.makedirs("features", exist_ok=True)

data = []

audio_extensions = (".wav", ".mp3", ".flac", ".ogg")

print("Starting feature extraction...")
print("Dataset:", DATASET_PATH)

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if not file.lower().endswith(audio_extensions):
            continue

        file_path = os.path.join(root, file)

        # Folder name becomes the label
        label = os.path.basename(root)

        try:

            # Load audio and convert to 16 kHz mono
            y, sr = librosa.load(
                file_path,
                sr=TARGET_SR,
                mono=True
            )

            # Skip empty audio
            if len(y) == 0:
                print("Skipped empty file:", file)
                continue

            # Normalize audio
            y = librosa.util.normalize(y)

            # -----------------------------
            # MFCC
            # -----------------------------

            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=N_MFCC
            )

            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            # -----------------------------
            # Mel Spectrogram
            # -----------------------------

            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=40
            )

            mel_db = librosa.power_to_db(
                mel,
                ref=np.max
            )

            mel_mean = np.mean(mel_db, axis=1)

            # -----------------------------
            # Zero Crossing Rate
            # -----------------------------

            zcr = librosa.feature.zero_crossing_rate(y)

            zcr_mean = np.mean(zcr)
            zcr_std = np.std(zcr)

            # -----------------------------
            # Spectral Centroid
            # -----------------------------

            centroid = librosa.feature.spectral_centroid(
                y=y,
                sr=sr
            )

            centroid_mean = np.mean(centroid)
            centroid_std = np.std(centroid)

            # -----------------------------
            # Spectral Bandwidth
            # -----------------------------

            bandwidth = librosa.feature.spectral_bandwidth(
                y=y,
                sr=sr
            )

            bandwidth_mean = np.mean(bandwidth)
            bandwidth_std = np.std(bandwidth)

            # -----------------------------
            # Spectral Rolloff
            # -----------------------------

            rolloff = librosa.feature.spectral_rolloff(
                y=y,
                sr=sr
            )

            rolloff_mean = np.mean(rolloff)
            rolloff_std = np.std(rolloff)

            # -----------------------------
            # RMS Energy
            # -----------------------------

            rms = librosa.feature.rms(y=y)

            rms_mean = np.mean(rms)
            rms_std = np.std(rms)

            # -----------------------------
            # Pitch
            # -----------------------------

            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=80,
                fmax=1000
            )

            valid_f0 = f0[~np.isnan(f0)]

            if len(valid_f0) > 0:

                pitch_mean = np.mean(valid_f0)
                pitch_std = np.std(valid_f0)
                pitch_min = np.min(valid_f0)
                pitch_max = np.max(valid_f0)

            else:

                pitch_mean = 0
                pitch_std = 0
                pitch_min = 0
                pitch_max = 0

            # -----------------------------
            # Create feature dictionary
            # -----------------------------

            features = {
                "filename": file,
                "filepath": file_path,
                "label": label,
                "duration": librosa.get_duration(y=y, sr=sr),
                "sample_rate": sr,

                "zcr_mean": zcr_mean,
                "zcr_std": zcr_std,

                "spectral_centroid_mean": centroid_mean,
                "spectral_centroid_std": centroid_std,

                "spectral_bandwidth_mean": bandwidth_mean,
                "spectral_bandwidth_std": bandwidth_std,

                "spectral_rolloff_mean": rolloff_mean,
                "spectral_rolloff_std": rolloff_std,

                "rms_mean": rms_mean,
                "rms_std": rms_std,

                "pitch_mean": pitch_mean,
                "pitch_std": pitch_std,
                "pitch_min": pitch_min,
                "pitch_max": pitch_max
            }

            # Add MFCC features
            for i in range(N_MFCC):

                features[f"mfcc_{i+1}_mean"] = mfcc_mean[i]
                features[f"mfcc_{i+1}_std"] = mfcc_std[i]

            # Add Mel features
            for i in range(40):

                features[f"mel_{i+1}_mean"] = mel_mean[i]

            data.append(features)

            print("Processed:", len(data), "|", label, "|", file)

        except Exception as e:

            print("\nERROR:")
            print(file_path)
            print(e)


# -----------------------------------------
# Create DataFrame
# -----------------------------------------

df = pd.DataFrame(data)

# Save CSV
df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n====================================")
print("FEATURE EXTRACTION COMPLETED")
print("====================================")

print("Total files processed:", len(df))

print("Number of features:", len(df.columns))

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nSaved to:")
print(OUTPUT_PATH)

print("\nCSV shape:")
print(df.shape)