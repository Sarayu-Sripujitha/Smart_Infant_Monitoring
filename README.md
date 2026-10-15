# Smart Infant Monitoring AI

An AI and IoT-based system for infant monitoring using infant cry analysis, environmental sensing, movement monitoring, and intelligent decision support.

## Current Research Progress

### WP1 — Infant Cry Dataset and Acoustic Phenotyping
Status: Completed

- Collected and organized 1,097 infant audio recordings
- Verified and standardized dataset labels
- Finalized 7 audio categories
- Standardized audio to 16 kHz mono
- Performed acoustic analysis
- Extracted MFCC and other acoustic features
- Generated an 85-feature dataset

### WP2 — AI-Based Infant Cry Pattern Recognition
Status: In Progress

The baseline machine-learning stage has been completed.

Models evaluated:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting

Current best baseline:
- Model: SVM
- Accuracy: 35.45%
- Macro F1-score: 39.08%

Further WP2 work will include:
- Feature selection and optimization
- Class imbalance handling
- Hyperparameter tuning
- Cross-validation
- Deep learning
- Uncertainty estimation
- Explainable AI
- Independent testing

## Dataset Categories

| Category | Samples |
|---|---:|
| Hungry | 382 |
| Discomfort | 135 |
| Tired | 132 |
| Belly Pain | 124 |
| Burping | 108 |
| Cold/Hot | 108 |
| Silence | 108 |
| **Total** | **1097** |

## Project Structure

```text
analysis/
features/
models/
preprocessing/
results/
dataset/
inspect_dataset.py
requirements.txt