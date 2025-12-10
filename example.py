"""
Example script demonstrating ECG-based glucose monitoring
"""

import numpy as np
import pandas as pd
from src.data_loader import SyntheticDataGenerator
from src.pipeline import GlucoseMonitoringPipeline, run_personalized_training


def example_with_synthetic_data():
    """Example using synthetic data"""
    
    print("=" * 60)
    print("ECG-Based Glucose Monitoring - Example")
    print("=" * 60)
    print()
    
    # Generate synthetic patient data (shorter duration for demo)
    print("1. Generating synthetic patient data...")
    generator = SyntheticDataGenerator(sampling_rate=250)
    patient_data = generator.generate_patient_data(duration_hours=2, patient_id='synthetic_001')
    
    print(f"   - ECG signal length: {len(patient_data['ecg_signal'])} samples")
    print(f"   - Sampling rate: {patient_data['sampling_rate']} Hz")
    print(f"   - Duration: {len(patient_data['ecg_signal']) / patient_data['sampling_rate'] / 3600:.1f} hours")
    print(f"   - CGM readings: {len(patient_data['cgm_values'])}")
    print(f"   - Glucose range: {patient_data['cgm_values'].min():.1f} - {patient_data['cgm_values'].max():.1f} mg/dL")
    print()
    
    # Train model for hypoglycemia detection
    print("2. Training personalized model for hypoglycemia detection...")
    print("   (This may take a few minutes...)")
    pipeline, metrics = run_personalized_training(
        patient_data,
        task='hypoglycemia',
        threshold=70
    )
    
    if metrics is not None:
        print()
        print("3. Evaluation Results:")
        print("   " + "-" * 40)
        for metric_name, value in metrics.items():
            print(f"   {metric_name:15s}: {value:.4f}")
        print("   " + "-" * 40)
    else:
        print("   Training failed. Check data quality.")
    
    print()
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


def example_feature_extraction():
    """Example demonstrating feature extraction"""
    
    print("\n" + "=" * 60)
    print("Feature Extraction Example")
    print("=" * 60)
    print()
    
    # Generate synthetic data
    generator = SyntheticDataGenerator(sampling_rate=250)
    patient_data = generator.generate_patient_data(duration_hours=1, patient_id='test_patient')
    
    # Initialize pipeline
    pipeline = GlucoseMonitoringPipeline(sampling_rate=250)
    
    # Extract features
    print("Extracting features from ECG signal...")
    features = pipeline.extract_features_from_ecg(patient_data['ecg_signal'])
    
    if features is not None:
        print(f"\nBeat-level morphology features:")
        print(f"  - Number of beats detected: {len(features['beat_features'])}")
        print(f"  - Features per beat: {features['beat_features'].shape[1]}")
        print(f"\nFeature names (first 10):")
        for i, col in enumerate(features['beat_features'].columns[:10]):
            print(f"  {i+1:2d}. {col}")
        
        print(f"\nHRV features:")
        print(f"  - Number of HRV features: {len(features['hrv_features'])}")
        if features['hrv_features']:
            print(f"  - HRV feature names (first 5):")
            for i, key in enumerate(list(features['hrv_features'].keys())[:5]):
                value = features['hrv_features'][key]
                print(f"     {i+1}. {key}: {value:.4f}")
    else:
        print("Feature extraction failed.")
    
    print()


def example_model_comparison():
    """Example comparing different models"""
    
    print("\n" + "=" * 60)
    print("Model Comparison Example")
    print("=" * 60)
    print()
    
    # Generate synthetic data
    generator = SyntheticDataGenerator(sampling_rate=250)
    patient_data = generator.generate_patient_data(duration_hours=2)
    
    print("Training and comparing different models...")
    print("(This demonstrates the different model architectures)")
    print()
    
    # Train beat-level model
    print("1. MBeat (Beat-level model)")
    pipeline_beat, metrics_beat = run_personalized_training(
        patient_data, task='hypoglycemia', threshold=70
    )
    
    if metrics_beat:
        print(f"   AUC: {metrics_beat['auc']:.4f}, F1: {metrics_beat['f1_score']:.4f}")
    print()
    
    # Train for hyperglycemia
    print("2. Training for hyperglycemia detection...")
    pipeline_hyper, metrics_hyper = run_personalized_training(
        patient_data, task='hyperglycemia', threshold=180
    )
    
    if metrics_hyper:
        print(f"   AUC: {metrics_hyper['auc']:.4f}, F1: {metrics_hyper['f1_score']:.4f}")
    print()
    
    print("Model comparison completed!")


if __name__ == "__main__":
    # Run examples
    try:
        # Main example with synthetic data
        example_with_synthetic_data()
        
        # Feature extraction example
        example_feature_extraction()
        
        # Model comparison example
        # example_model_comparison()  # Uncomment to run
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
