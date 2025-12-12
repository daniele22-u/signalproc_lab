# ECG-Based Glucose Level Monitoring

Implementation of the model from the paper: "Glucose level monitoring by ECG" (https://doi.org/10.1016/j.bspc.2024.106569)

## Overview

This project implements a machine learning pipeline for detecting hypoglycemia and hyperglycemia using ECG signals. The implementation follows the methodology described in the paper and includes:

- ECG signal processing and fiducial point detection
- Morphology feature extraction (35 features)
- HRV time-domain feature extraction (18 features)
- Multiple model architectures:
  - **MBeat**: Beat-level Random Forest model
  - **MHRV**: HRV-based interval model
  - **MMorph**: Morphology aggregation model
  - **MMorph+HRV**: Combined features model
  - **Fusion Model**: Multi-threshold ensemble model

## Dataset

The implementation is designed to work with the D1NAMO ECG-Glucose dataset:
- **Dataset URL**: https://www.kaggle.com/datasets/sarabhian/d1namo-ecg-glucose-data
- **ECG**: 250 Hz sampling rate (Zephyr Bioharness)
- **CGM**: 5-minute intervals (Dexcom CGM)

## Installation

### Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download the D1NAMO dataset from Kaggle
# Place the data in data/raw/ directory
```

## Project Structure

```
signalproc_lab/
├── src/
│   ├── __init__.py
│   ├── signal_processing.py      # ECG preprocessing and fiducial detection
│   ├── feature_extraction.py     # Morphology and HRV features
│   ├── models.py                 # Model implementations
│   ├── data_loader.py            # Data loading utilities
│   └── pipeline.py               # End-to-end training pipeline
├── data/
│   ├── raw/                      # Raw dataset (not in git)
│   └── processed/                # Processed data (not in git)
├── models/                       # Saved models (not in git)
├── notebooks/                    # Jupyter notebooks
├── requirements.txt
└── README.md
```

## Usage

### 🚀 Quick Start with Interactive Notebook (Recommended)

**NEW!** Try the comprehensive Jupyter notebook for an interactive learning experience:

```bash
# Install Jupyter if not already installed
pip install jupyter

# Open the tutorial notebook
jupyter notebook tutorial_completo.ipynb
```

The notebook (`tutorial_completo.ipynb`) provides:
- ✅ **Complete tutorial in Italian** with step-by-step explanations
- ✅ **Interactive code cells** - run and experiment in real-time
- ✅ **Visualizations** - plots and graphs inline
- ✅ **Works out-of-the-box** - uses synthetic data (no dataset required)
- ✅ **All features covered** - from data loading to model evaluation

See [NOTEBOOK_README.md](NOTEBOOK_README.md) for detailed notebook documentation.

### Quick Start with Example Script

```python
from src.data_loader import SyntheticDataGenerator
from src.pipeline import run_personalized_training

# Generate synthetic data (for testing without real dataset)
generator = SyntheticDataGenerator(sampling_rate=250)
patient_data = generator.generate_patient_data(duration_hours=48)

# Train personalized model for hypoglycemia detection
pipeline, metrics = run_personalized_training(
    patient_data, 
    task='hypoglycemia', 
    threshold=70
)

# Print evaluation metrics
print("Model Performance:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")
```

### Step-by-Step Usage

```python
from src.signal_processing import ECGProcessor
from src.feature_extraction import MorphologyFeatureExtractor, HRVFeatureExtractor
from src.models import MBeat
from src.data_loader import D1NAMODataLoader

# 1. Load data
loader = D1NAMODataLoader(data_dir='data/raw')
patient_data = loader.load_patient_data(patient_id='001')

# 2. Process ECG signal
processor = ECGProcessor(sampling_rate=250)
cleaned_ecg = processor.preprocess(patient_data['ecg_signal'])
fiducial_points = processor.detect_fiducial_points(cleaned_ecg)

# 3. Extract features
morph_extractor = MorphologyFeatureExtractor(sampling_rate=250)
features = []
for beat_idx in range(len(fiducial_points['R_peaks'])):
    beat_features = morph_extractor.extract_beat_features(
        cleaned_ecg, fiducial_points, beat_idx
    )
    if beat_features is not None:
        features.append(beat_features)

# 4. Train model
import pandas as pd
X = pd.DataFrame(features).fillna(0).values
y = (patient_data['aligned_glucose'] < 70).astype(int)  # Hypoglycemia labels

model = MBeat(n_estimators=100, random_state=42)
model.train(X, y)

# 5. Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"AUC: {metrics['auc']:.4f}")
```

## Models

### MBeat (Beat-Level Model)
- Input: 35 morphology features per beat
- Output: Probability of hypoglycemia/hyperglycemia for each beat
- Algorithm: Random Forest classifier

### MHRV (HRV Model)
- Input: 18 HRV time-domain features per 1-minute interval
- Output: Interval-level prediction
- Algorithm: Random Forest classifier

### MMorph+HRV (Combined Model)
- Input: Aggregated morphology features + HRV features
- Output: Interval-level prediction
- Algorithm: Random Forest classifier

### Fusion Model
- Multiple beat-level models trained at different glucose thresholds
- Hypoglycemia thresholds: [55, 60, 65, 70, 75, 80, 85, 90] mg/dL
- Hyperglycemia thresholds: [150, 165, 180, 200, 225, 250] mg/dL
- Combines predictions from all threshold models
- Final Random Forest classifier for interval-level prediction

## Features

### Morphology Features (35 total)
1. **Amplitudes (5)**: P, Q, R, S, T peak amplitudes
2. **Intervals (10)**: PR, QR, RS, RT, QS, QT, ST, PS, PT, PQ intervals
3. **Euclidean Distances (9)**: Between fiducial points
4. **Slopes (9)**: Between fiducial points
5. **Additional (2)**: RR interval, Heart Rate

### HRV Features (18 time-domain)
Computed using NeuroKit2 package:
- SDNN, RMSSD, SDSD, NN50, pNN50, NN20, pNN20
- HTI, TINN, Mean NN, Median NN, Range NN, IQR
- And more...

## Evaluation Metrics

- **AUC**: Area Under ROC Curve
- **Sensitivity (Recall)**: True Positive Rate
- **Specificity**: True Negative Rate
- **Precision**: Positive Predictive Value
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall correctness

## Clinical Thresholds

- **Hypoglycemia**: < 70 mg/dL
- **Hyperglycemia**: > 180 mg/dL
- **Normal (Euglycemia)**: 70-180 mg/dL

## Personalized Modeling

The implementation follows a personalized approach:
- Separate models trained for each patient
- Accounts for inter-subject variability in ECG features
- Temporal validation: Last 24 hours used for testing

## References

1. Main paper: D. Dave et al., "Glucose level monitoring by ECG," Biomedical Signal Processing and Control, vol. 96, 2024.
   - DOI: https://doi.org/10.1016/j.bspc.2024.106569

2. D1NAMO Dataset: https://www.kaggle.com/datasets/sarabhian/d1namo-ecg-glucose-data

## License

This implementation is for educational and research purposes.

## Contact

For questions or issues, please open an issue on the repository.
