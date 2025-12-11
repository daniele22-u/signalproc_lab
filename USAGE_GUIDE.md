# Usage Guide: ECG-Based Glucose Monitoring

This guide explains how to use the implementation with the D1NAMO dataset and replicate the results from the paper.

## Prerequisites

1. **Download the D1NAMO Dataset**
   - URL: https://www.kaggle.com/datasets/sarabhian/d1namo-ecg-glucose-data
   - You'll need a Kaggle account to download the dataset
   - Place the downloaded data in `data/raw/` directory

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure

The D1NAMO dataset should be organized as follows:
```
data/raw/
├── patient_001/
│   ├── ecg.csv          # ECG data at 250 Hz
│   └── cgm.csv          # CGM data at 5-minute intervals
├── patient_002/
│   ├── ecg.csv
│   └── cgm.csv
...
```

Expected CSV format:
- **ecg.csv**: columns `[timestamp, ecg]`
- **cgm.csv**: columns `[timestamp, glucose]`

## Quick Start

### 1. Test with Synthetic Data (No Dataset Required)

```python
from src.data_loader import SyntheticDataGenerator
from src.pipeline import run_personalized_training

# Generate synthetic data
generator = SyntheticDataGenerator(sampling_rate=250)
patient_data = generator.generate_patient_data(duration_hours=48)

# Train model
pipeline, metrics = run_personalized_training(
    patient_data, 
    task='hypoglycemia', 
    threshold=70
)

print("Metrics:", metrics)
```

### 2. Use Real D1NAMO Dataset

```python
from src.data_loader import D1NAMODataLoader
from src.pipeline import run_personalized_training

# Load patient data
loader = D1NAMODataLoader(data_dir='data/raw')
patient_data = loader.load_patient_data(patient_id='001')

# Train personalized model
pipeline, metrics = run_personalized_training(
    patient_data, 
    task='hypoglycemia', 
    threshold=70
)

print("Performance Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")
```

## Complete Workflow

### Step 1: Data Loading and Preprocessing

```python
from src.data_loader import D1NAMODataLoader

# Initialize loader
loader = D1NAMODataLoader(data_dir='data/raw')

# Load patient data
patient_data = loader.load_patient_data(patient_id='001')

# Check data quality
print(f"ECG samples: {len(patient_data['ecg_signal'])}")
print(f"CGM readings: {len(patient_data['cgm_values'])}")
```

### Step 2: Temporal Train-Test Split

```python
# Split data (last 24 hours for testing)
train_data, test_data = loader.temporal_train_test_split(
    patient_data, 
    test_hours=24
)
```

### Step 3: Feature Extraction

```python
from src.pipeline import GlucoseMonitoringPipeline

# Initialize pipeline
pipeline = GlucoseMonitoringPipeline(sampling_rate=250)

# Extract features
train_features = pipeline.extract_features_from_ecg(
    train_data['ecg_signal']
)

test_features = pipeline.extract_features_from_ecg(
    test_data['ecg_signal']
)
```

### Step 4: Model Training

#### Beat-Level Model (MBeat)

```python
from src.models import MBeat

# Prepare training data
X_train, y_train = pipeline.prepare_beat_level_data(
    train_features,
    train_data['cgm_values'],
    threshold=70,
    task='hypoglycemia'
)

# Train model
model = MBeat(n_estimators=100, random_state=42)
model.train(X_train, y_train)

# Save model
model.save('models/mbeat_patient001_hypo.pkl')
```

#### HRV-Based Model (MHRV)

```python
from src.models import MHRV

# Extract HRV features for intervals
hrv_features = train_features['hrv_features']

# Train HRV model
model_hrv = MHRV(n_estimators=100, random_state=42)
model_hrv.train(X_hrv_train, y_interval_train)
```

#### Combined Model (MMorph+HRV)

```python
from src.models import MMorphHRV

# Combine morphology and HRV features
X_combined = np.hstack([X_morph, X_hrv])

# Train combined model
model_combined = MMorphHRV(n_estimators=100, random_state=42)
model_combined.train(X_combined, y_train)
```

#### Fusion Model

```python
from src.models import FusionModel

# Initialize fusion model
fusion_model = FusionModel(
    thresholds_hypo=[55, 60, 65, 70, 75, 80, 85, 90],
    n_estimators=100,
    random_state=42
)

# Train beat-level models at multiple thresholds
fusion_model.train_beat_models(
    X_beats_train, 
    glucose_train, 
    task='hypoglycemia'
)

# Get fusion features
fusion_features = fusion_model.get_fusion_features(
    X_beats_train, 
    task='hypoglycemia'
)

# Train interval-level fusion model
X_interval = np.hstack([fusion_features, hrv_features])
fusion_model.train_interval_model(X_interval, y_interval)
```

### Step 5: Evaluation

```python
# Prepare test data
X_test, y_test = pipeline.prepare_beat_level_data(
    test_features,
    test_data['cgm_values'],
    threshold=70,
    task='hypoglycemia'
)

# Evaluate model
metrics = model.evaluate(X_test, y_test)

print("\nEvaluation Results:")
print(f"AUC:         {metrics['auc']:.4f}")
print(f"Sensitivity: {metrics['sensitivity']:.4f}")
print(f"Specificity: {metrics['specificity']:.4f}")
print(f"Precision:   {metrics['precision']:.4f}")
print(f"F1-Score:    {metrics['f1_score']:.4f}")
```

## Training for All Patients (Personalized Models)

```python
from pathlib import Path

# Get all patient IDs
data_dir = Path('data/raw')
patient_ids = [p.name for p in data_dir.iterdir() if p.is_dir()]

# Train personalized model for each patient
results = {}

for patient_id in patient_ids:
    print(f"\nTraining model for patient {patient_id}...")
    
    # Load data
    patient_data = loader.load_patient_data(patient_id)
    
    # Train model
    pipeline, metrics = run_personalized_training(
        patient_data,
        task='hypoglycemia',
        threshold=70
    )
    
    # Store results
    results[patient_id] = metrics
    
    # Save model
    pipeline.save_model('MBeat', f'models/mbeat_{patient_id}_hypo.pkl')

# Analyze results across all patients
import pandas as pd
results_df = pd.DataFrame(results).T
print("\nAverage Performance Across All Patients:")
print(results_df.mean())
```

## Hyperglycemia Detection

To train models for hyperglycemia detection instead:

```python
# Change task and threshold
pipeline, metrics = run_personalized_training(
    patient_data,
    task='hyperglycemia',  # Changed from 'hypoglycemia'
    threshold=180          # Changed from 70
)
```

## Model Comparison

```python
# Train different models and compare
models_to_compare = ['MBeat', 'MHRV', 'MMorph+HRV', 'Fusion']
comparison_results = {}

for model_name in models_to_compare:
    print(f"\nTraining {model_name}...")
    
    # Train model (implement specific training for each)
    # ... training code ...
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    comparison_results[model_name] = metrics

# Display comparison
import pandas as pd
comparison_df = pd.DataFrame(comparison_results).T
print("\nModel Comparison:")
print(comparison_df)
```

## Tips and Best Practices

1. **Data Quality**: Ensure ECG and CGM data are properly synchronized
2. **Temporal Validation**: Always use temporal splits (not random) to avoid data leakage
3. **Personalization**: Train separate models for each patient due to inter-subject variability
4. **Class Imbalance**: Handle class imbalance if hypoglycemia/hyperglycemia events are rare
5. **Feature Scaling**: Random Forest doesn't require feature scaling, but consider it for other algorithms
6. **Cross-Validation**: Use temporal cross-validation with multiple test periods

## Troubleshooting

### Issue: "Failed to extract features"
- **Cause**: ECG signal too short or poor quality
- **Solution**: Ensure at least 1-2 minutes of continuous ECG data

### Issue: "Model evaluation shows all zeros"
- **Cause**: No positive class examples in test set
- **Solution**: Ensure test set has sufficient hypoglycemia/hyperglycemia events

### Issue: "Memory error during training"
- **Cause**: Processing very long ECG signals
- **Solution**: Process data in chunks or use a machine with more RAM

## References

1. Paper: D. Dave et al., "Glucose level monitoring by ECG," Biomedical Signal Processing and Control, 2024
2. Dataset: D1NAMO ECG-Glucose Data - https://www.kaggle.com/datasets/sarabhian/d1namo-ecg-glucose-data
3. NeuroKit2 Documentation: https://neuropsychology.github.io/NeuroKit/

## Support

For issues or questions:
1. Check this guide first
2. Review the example.py script
3. Explore the demo.ipynb notebook
4. Open an issue on the repository
