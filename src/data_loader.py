"""
Data Loading and Preprocessing Module
Handles loading and preprocessing of D1NAMO dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


class D1NAMODataLoader:
    """Load and preprocess D1NAMO ECG-Glucose dataset"""
    
    def __init__(self, data_dir):
        """
        Initialize data loader
        
        Args:
            data_dir: Path to directory containing D1NAMO dataset
        """
        self.data_dir = Path(data_dir)
    
    def load_patient_data(self, patient_id):
        """
        Load ECG and CGM data for a specific patient
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dictionary containing ECG signal, timestamps, and CGM readings
        """
        # This is a placeholder implementation
        # The actual implementation depends on the D1NAMO dataset format
        
        patient_data = {
            'patient_id': patient_id,
            'ecg_signal': None,  # Shape: (n_samples,)
            'ecg_timestamps': None,  # Shape: (n_samples,)
            'cgm_values': None,  # Shape: (n_cgm_readings,)
            'cgm_timestamps': None,  # Shape: (n_cgm_readings,)
            'sampling_rate': 250  # Hz
        }
        
        # Load ECG data
        ecg_file = self.data_dir / f"patient_{patient_id}_ecg.csv"
        if ecg_file.exists():
            try:
                ecg_df = pd.read_csv(ecg_file)
                if 'ecg' not in ecg_df.columns or 'timestamp' not in ecg_df.columns:
                    raise ValueError(f"ECG file must contain 'ecg' and 'timestamp' columns")
                patient_data['ecg_signal'] = ecg_df['ecg'].values
                patient_data['ecg_timestamps'] = pd.to_datetime(ecg_df['timestamp'])
            except Exception as e:
                print(f"Error loading ECG data: {e}")
                return patient_data
        
        # Load CGM data
        cgm_file = self.data_dir / f"patient_{patient_id}_cgm.csv"
        if cgm_file.exists():
            try:
                cgm_df = pd.read_csv(cgm_file)
                if 'glucose' not in cgm_df.columns or 'timestamp' not in cgm_df.columns:
                    raise ValueError(f"CGM file must contain 'glucose' and 'timestamp' columns")
                patient_data['cgm_values'] = cgm_df['glucose'].values
                patient_data['cgm_timestamps'] = pd.to_datetime(cgm_df['timestamp'])
            except Exception as e:
                print(f"Error loading CGM data: {e}")
        
        return patient_data
    
    def align_ecg_with_cgm(self, ecg_timestamps, cgm_timestamps, cgm_values):
        """
        Align ECG beats with nearest CGM readings (forward direction)
        
        Args:
            ecg_timestamps: Timestamps for ECG beats
            cgm_timestamps: Timestamps for CGM readings
            cgm_values: Glucose values from CGM
            
        Returns:
            Array of glucose values aligned with ECG beats
        """
        aligned_glucose = np.zeros(len(ecg_timestamps))
        
        for i, ecg_time in enumerate(ecg_timestamps):
            # Find nearest CGM reading in forward direction
            future_readings = cgm_timestamps >= ecg_time
            if np.any(future_readings):
                nearest_idx = np.where(future_readings)[0][0]
                aligned_glucose[i] = cgm_values[nearest_idx]
            else:
                # If no future reading, use the last available
                aligned_glucose[i] = cgm_values[-1]
        
        return aligned_glucose
    
    def temporal_train_test_split(self, data, test_hours=24, test_ratio=0.2):
        """
        Temporal train-test split (last portion for testing)
        
        Args:
            data: Patient data dictionary with timestamps
            test_hours: Number of hours for test set (default: 24)
            test_ratio: Ratio for test set if test_hours is too large (default: 0.2)
            
        Returns:
            train_data, test_data dictionaries
        """
        if data['ecg_timestamps'] is None:
            return None, None
        
        # Get the timestamp boundary
        last_timestamp = data['ecg_timestamps'].max()
        first_timestamp = data['ecg_timestamps'].min()
        total_duration = (last_timestamp - first_timestamp).total_seconds() / 3600  # hours
        
        # If test_hours is too large, use ratio instead
        if test_hours > total_duration * 0.5:
            test_hours = total_duration * test_ratio
        
        split_timestamp = last_timestamp - timedelta(hours=test_hours)
        
        # Split ECG data
        train_mask = data['ecg_timestamps'] < split_timestamp
        test_mask = data['ecg_timestamps'] >= split_timestamp
        
        train_data = {
            'patient_id': data['patient_id'],
            'ecg_signal': data['ecg_signal'][train_mask],
            'ecg_timestamps': data['ecg_timestamps'][train_mask],
            'sampling_rate': data['sampling_rate']
        }
        
        test_data = {
            'patient_id': data['patient_id'],
            'ecg_signal': data['ecg_signal'][test_mask],
            'ecg_timestamps': data['ecg_timestamps'][test_mask],
            'sampling_rate': data['sampling_rate']
        }
        
        # Split CGM data if available
        if data['cgm_timestamps'] is not None:
            cgm_train_mask = data['cgm_timestamps'] < split_timestamp
            cgm_test_mask = data['cgm_timestamps'] >= split_timestamp
            
            train_data['cgm_values'] = data['cgm_values'][cgm_train_mask]
            train_data['cgm_timestamps'] = data['cgm_timestamps'][cgm_train_mask]
            
            test_data['cgm_values'] = data['cgm_values'][cgm_test_mask]
            test_data['cgm_timestamps'] = data['cgm_timestamps'][cgm_test_mask]
        
        return train_data, test_data
    
    def create_intervals(self, timestamps, interval_duration=60):
        """
        Create non-overlapping time intervals
        
        Args:
            timestamps: Array of timestamps
            interval_duration: Duration of each interval in seconds (default: 60)
            
        Returns:
            Array of interval IDs for each timestamp
        """
        if len(timestamps) == 0:
            return np.array([])
        
        # Convert to seconds from start
        start_time = timestamps.min()
        seconds_from_start = (timestamps - start_time).total_seconds()
        
        # Assign interval IDs
        interval_ids = (seconds_from_start // interval_duration).astype(int)
        
        return interval_ids


class SyntheticDataGenerator:
    """Generate synthetic data for testing when real data is not available"""
    
    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate
    
    def generate_patient_data(self, duration_hours=48, patient_id='synthetic_001'):
        """
        Generate synthetic ECG and CGM data
        
        Args:
            duration_hours: Duration of data in hours
            patient_id: Patient identifier
            
        Returns:
            Dictionary containing synthetic data
        """
        # Generate time arrays
        n_samples = duration_hours * 3600 * self.sampling_rate
        ecg_timestamps = pd.date_range(start='2024-01-01', periods=n_samples, 
                                       freq=f'{1000/self.sampling_rate}ms')
        
        # Generate synthetic ECG with some variation
        t = np.arange(n_samples) / self.sampling_rate
        # Simulate heartbeats at ~70 bpm with R-peaks
        heart_rate = 70 / 60  # beats per second
        ecg_signal = np.sin(2 * np.pi * heart_rate * t)
        # Add R-peaks
        r_peak_period = int(self.sampling_rate / heart_rate)
        r_peak_width = int(0.04 * self.sampling_rate)  # 40ms R-peak width
        for i in range(0, n_samples, r_peak_period):
            if i < n_samples:
                ecg_signal[i:min(i+r_peak_width, n_samples)] += 2.0
        # Add noise
        ecg_signal += np.random.normal(0, 0.1, n_samples)
        
        # Generate CGM data (every 5 minutes)
        n_cgm_samples = duration_hours * 12  # 12 readings per hour
        cgm_timestamps = pd.date_range(start='2024-01-01', periods=n_cgm_samples, freq='5min')
        
        # Simulate glucose variation (70-180 mg/dL normal, with some excursions)
        cgm_values = 120 + 30 * np.sin(2 * np.pi * np.arange(n_cgm_samples) / 48)  # Daily pattern
        cgm_values += np.random.normal(0, 10, n_cgm_samples)  # Add noise
        
        # Add some hypoglycemic and hyperglycemic episodes
        hypo_start = int(0.3 * n_cgm_samples)
        hypo_end = int(0.35 * n_cgm_samples)
        cgm_values[hypo_start:hypo_end] = 60 + np.random.normal(0, 5, hypo_end - hypo_start)
        
        hyper_start = int(0.6 * n_cgm_samples)
        hyper_end = int(0.65 * n_cgm_samples)
        cgm_values[hyper_start:hyper_end] = 200 + np.random.normal(0, 10, hyper_end - hyper_start)
        
        return {
            'patient_id': patient_id,
            'ecg_signal': ecg_signal,
            'ecg_timestamps': ecg_timestamps,
            'cgm_values': cgm_values,
            'cgm_timestamps': cgm_timestamps,
            'sampling_rate': self.sampling_rate
        }
