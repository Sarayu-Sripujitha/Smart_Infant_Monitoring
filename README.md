# Smart Infant Monitoring AI 👶

An AI- and IoT-based infant monitoring system that aims to analyze infant cries, monitor environmental conditions, track movement, and support caregivers through intelligent insights.

## Research Progress

### WP1: Infant Cry Dataset and Acoustic Phenotyping

**Status: Completed ✅**

WP1 focuses on inspecting the infant cry dataset, preprocessing audio recordings, analyzing acoustic characteristics, and extracting features for machine-learning research.

### Objectives

- Inspect and organize infant audio recordings.
- Verify dataset labels and category distribution.
- Standardize audio recordings to 16 kHz mono.
- Analyze audio waveforms and spectrograms.
- Extract MFCC and other acoustic features.
- Prepare a structured feature dataset for machine-learning experiments.

## Dataset Overview

- **Total audio recordings:** 1,097
- **Number of categories:** 7
- **Audio format:** Audio recordings processed as mono
- **Standardized sample rate:** 16,000 Hz
- **Average audio duration:** Approximately 6.6 seconds
- **Extracted acoustic features:** 85 columns

### Dataset Categories

| Category | Number of Samples |
|---|---:|
| Hungry | 382 |
| Discomfort | 135 |
| Tired | 132 |
| Belly Pain | 124 |
| Burping | 108 |
| Cold/Hot | 108 |
| Silence | 108 |
| **Total** | **1,097** |

## Acoustic Feature Extraction

The project extracts acoustic features from infant audio recordings to support future cry classification.

Features include:

- **MFCC:** Mel-Frequency Cepstral Coefficients
- **Mel Features:** Mel-scale spectral characteristics
- **Zero-Crossing Rate:** Measures changes in the audio signal's sign.
- **Spectral Centroid:** Estimates the spectral centre of mass.
- **Spectral Bandwidth:** Measures the spread of spectral energy.
- **Spectral Rolloff:** Describes the frequency below which a specified proportion of spectral energy lies.
- **RMS Energy:** Represents the signal's average energy level.
- **Pitch Statistics:** Describes estimated pitch characteristics.

The extracted features are stored in `features/features.csv`.

## Visual Analysis

The following visualizations were generated during WP1:

- **Class Distribution:** Shows the number of recordings in each category.
- **Waveform:** Displays changes in audio amplitude over time.
- **Mel Spectrogram:** Represents the distribution of frequency energy over time.
- **MFCC Visualization:** Shows acoustic patterns represented by MFCC coefficients.

## Project Structure

```text
Smart_Infant_AI/
├── analysis/
│   ├── acoustic_features.py
│   ├── audio_analysis.py
│   └── class_distribution.py
├── features/
│   └── features.csv
├── preprocessing/
│   └── extract_features.py
├── results/
│   ├── class_distribution.png
│   ├── mel_spectrogram.png
│   ├── mfcc.png
│   └── waveform.png
├── inspect_dataset.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies Used

- Python
- Librosa
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SoundFile

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Sarayu-Sripujitha/Smart_Infant_Monitoring.git
```

### 2. Navigate to the Project Directory

```bash
cd Smart_Infant_Monitoring
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Place the dataset in the appropriate local `dataset/` directory before running the analysis scripts. The raw audio dataset is not included in this repository.

### 5. Run Dataset Inspection

```bash
python inspect_dataset.py
```

Run the analysis and preprocessing scripts as needed to reproduce the WP1 results.

## Current Scope

This repository currently covers **WP1: Infant Cry Dataset and Acoustic Phenotyping**.

The raw audio dataset and WP2 model-training files are maintained separately and are not included in this upload.

## Future Work

- Improve infant cry pattern recognition.
- Explore feature selection and optimization.
- Evaluate machine-learning and deep-learning models.
- Investigate explainable AI and prediction confidence.
- Integrate cry analysis with environmental and movement sensor data.
- Develop an integrated infant monitoring prototype.

## Disclaimer

This project is intended for academic research and prototype development. It is not a substitute for professional medical advice, diagnosis, or monitoring.

## Author

Developed as part of an academic research project on AI-based infant monitoring.