"""
Training and Prediction Pipeline
Ties together all components for end-to-end workflow
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from .signal_processing import ECGProcessor
from .feature_extraction import MorphologyFeatureExtractor, HRVFeatureExtractor, IntervalFeatureExtractor
from .models import MBeat, MHRV, MMorph, MMorphHRV, FusionModel, MajorityVotingModel
from .data_loader import D1NAMODataLoader


class GlucoseMonitoringPipeline:
    """Complete pipeline for ECG-based glucose monitoring"""
    
    def __init__(self, sampling_rate=250):
        """
        Initialize pipeline
        
        Args:
            sampling_rate: ECG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.ecg_processor = ECGProcessor(sampling_rate)
        self.morph_extractor = MorphologyFeatureExtractor(sampling_rate)
        self.hrv_extractor = HRVFeatureExtractor(sampling_rate)
        self.interval_extractor = IntervalFeatureExtractor()
        
        # Models
        self.models = {}
    
    def extract_features_from_ecg(self, ecg_signal, glucose_values=None):
        """
        Extract all features from ECG signal
        
        Args:
            ecg_signal: Raw ECG signal
            glucose_values: Corresponding glucose values (optional, for alignment)
            
        Returns:
            Dictionary containing beat-level and interval-level features
        """
        # Preprocess ECG
        cleaned_ecg = self.ecg_processor.preprocess(ecg_signal)
        
        # Detect fiducial points
        fiducial_points = self.ecg_processor.detect_fiducial_points(cleaned_ecg)
        r_peaks = fiducial_points['R_peaks']
        
        if len(r_peaks) < 2:
            return None
        
        # Extract beat-level morphology features
        beat_features_list = []
        for beat_idx in range(len(r_peaks)):
            beat_features = self.morph_extractor.extract_beat_features(
                cleaned_ecg, fiducial_points, beat_idx
            )
            if beat_features is not None:
                beat_features_list.append(beat_features)
        
        if len(beat_features_list) == 0:
            return None
        
        beat_features_df = pd.DataFrame(beat_features_list)
        
        # Extract HRV features for intervals (1-minute windows)
        # For simplicity, compute HRV for the entire segment
        hrv_features = self.hrv_extractor.extract_hrv_features(r_peaks)
        
        return {
            'beat_features': beat_features_df,
            'hrv_features': hrv_features,
            'r_peaks': r_peaks,
            'glucose_values': glucose_values
        }
    
    def prepare_beat_level_data(self, features, glucose_values, threshold=70, task='hypoglycemia'):
        """
        Prepare beat-level training data
        
        Args:
            features: Dictionary from extract_features_from_ecg
            glucose_values: Glucose values aligned with beats
            threshold: Glucose threshold for classification
            task: 'hypoglycemia' or 'hyperglycemia'
            
        Returns:
            X, y arrays for training
        """
        beat_features_df = features['beat_features']
        
        # Remove rows with NaN values
        X = beat_features_df.fillna(beat_features_df.mean()).values
        
        # Create labels
        if task == 'hypoglycemia':
            y = (glucose_values < threshold).astype(int)
        else:  # hyperglycemia
            y = (glucose_values > threshold).astype(int)
        
        # Ensure same length
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        return X, y
    
    def train_beat_model(self, X_train, y_train, model_name='MBeat'):
        """
        Train beat-level model
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name for the model
            
        Returns:
            Trained model
        """
        model = MBeat(n_estimators=100, random_state=42)
        model.train(X_train, y_train)
        self.models[model_name] = model
        return model
    
    def train_hrv_model(self, X_train, y_train, model_name='MHRV'):
        """
        Train HRV-based interval model
        
        Args:
            X_train: Training HRV features
            y_train: Training labels
            model_name: Name for the model
            
        Returns:
            Trained model
        """
        model = MHRV(n_estimators=100, random_state=42)
        model.train(X_train, y_train)
        self.models[model_name] = model
        return model
    
    def train_combined_model(self, X_train, y_train, model_name='MMorph+HRV'):
        """
        Train combined morphology + HRV model
        
        Args:
            X_train: Training features (morphology + HRV)
            y_train: Training labels
            model_name: Name for the model
            
        Returns:
            Trained model
        """
        model = MMorphHRV(n_estimators=100, random_state=42)
        model.train(X_train, y_train)
        self.models[model_name] = model
        return model
    
    def train_fusion_model(self, X_beats_train, glucose_train, 
                          X_interval_train, y_interval_train,
                          task='hypoglycemia', model_name='Fusion'):
        """
        Train fusion model
        
        Args:
            X_beats_train: Beat-level features for training
            glucose_train: Glucose values for beats
            X_interval_train: Interval-level features (HRV, etc.)
            y_interval_train: Interval-level labels
            task: 'hypoglycemia' or 'hyperglycemia'
            model_name: Name for the model
            
        Returns:
            Trained fusion model
        """
        # Create fusion model
        fusion_model = FusionModel(n_estimators=100, random_state=42)
        
        # Train beat-level models at different thresholds
        fusion_model.train_beat_models(X_beats_train, glucose_train, task=task)
        
        # Get fusion features for training
        fusion_features = fusion_model.get_fusion_features(X_beats_train, task=task)
        
        # Aggregate fusion features at interval level
        # This is a simplified version - in practice, you'd need to aggregate by intervals
        X_combined = np.hstack([fusion_features, X_interval_train])
        
        # Train interval-level model
        fusion_model.train_interval_model(X_combined, y_interval_train)
        
        self.models[model_name] = fusion_model
        return fusion_model
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a trained model
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train it first.")
        
        model = self.models[model_name]
        metrics = model.evaluate(X_test, y_test)
        
        return metrics
    
    def predict(self, model_name, X):
        """
        Make predictions using a trained model
        
        Args:
            model_name: Name of the model
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train it first.")
        
        model = self.models[model_name]
        return model.predict(X)
    
    def save_model(self, model_name, filepath):
        """Save a trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.models[model_name]
        model.save(filepath)
    
    def load_model(self, model_name, filepath):
        """Load a trained model"""
        if model_name == 'MBeat':
            model = MBeat()
        elif model_name == 'MHRV':
            model = MHRV()
        elif model_name == 'MMorph':
            model = MMorph()
        elif model_name == 'MMorph+HRV':
            model = MMorphHRV()
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        model.load(filepath)
        self.models[model_name] = model


def run_personalized_training(patient_data, task='hypoglycemia', threshold=70):
    """
    Run personalized model training for a single patient
    
    Args:
        patient_data: Dictionary containing patient's ECG and CGM data
        task: 'hypoglycemia' or 'hyperglycemia'
        threshold: Glucose threshold for classification
        
    Returns:
        Trained pipeline and evaluation metrics
    """
    pipeline = GlucoseMonitoringPipeline(sampling_rate=patient_data['sampling_rate'])
    
    # Split data temporally
    data_loader = D1NAMODataLoader(data_dir='.')
    train_data, test_data = data_loader.temporal_train_test_split(patient_data, test_hours=24)
    
    if train_data is None or test_data is None:
        print("Insufficient data for training/testing")
        return None, None
    
    # Extract features for training
    print("Extracting training features...")
    train_features = pipeline.extract_features_from_ecg(
        train_data['ecg_signal'], 
        glucose_values=train_data.get('cgm_values')
    )
    
    if train_features is None:
        print("Failed to extract training features")
        return None, None
    
    # Align glucose values with beats
    if 'cgm_values' in train_data and train_data['cgm_values'] is not None:
        # For simplicity, assume glucose values are pre-aligned
        glucose_train = train_data['cgm_values'][:len(train_features['beat_features'])]
    else:
        print("No CGM data available")
        return None, None
    
    # Prepare training data
    X_train, y_train = pipeline.prepare_beat_level_data(
        train_features, glucose_train, threshold=threshold, task=task
    )
    
    # Train beat-level model
    print(f"Training beat-level model for {task}...")
    model = pipeline.train_beat_model(X_train, y_train)
    
    # Extract features for testing
    print("Extracting test features...")
    test_features = pipeline.extract_features_from_ecg(
        test_data['ecg_signal'],
        glucose_values=test_data.get('cgm_values')
    )
    
    if test_features is None:
        print("Failed to extract test features")
        return pipeline, None
    
    glucose_test = test_data['cgm_values'][:len(test_features['beat_features'])]
    X_test, y_test = pipeline.prepare_beat_level_data(
        test_features, glucose_test, threshold=threshold, task=task
    )
    
    # Evaluate
    print("Evaluating model...")
    metrics = pipeline.evaluate_model('MBeat', X_test, y_test)
    
    return pipeline, metrics
