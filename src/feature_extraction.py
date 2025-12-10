"""
Feature Extraction Module
Extracts morphology and HRV features from ECG signals
"""

import numpy as np
import pandas as pd
import neurokit2 as nk


class MorphologyFeatureExtractor:
    """Extract morphology features from ECG beats"""
    
    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate
    
    def extract_beat_features(self, ecg_signal, fiducial_points, beat_idx):
        """
        Extract 35 morphology features for a single beat
        
        Features include:
        - 9 Euclidean distances between peaks
        - 10 interval-based distances
        - 5 peak amplitudes
        - 9 slopes between peaks
        - RR interval and HR
        
        Args:
            ecg_signal: ECG signal array
            fiducial_points: Dictionary with fiducial point indices
            beat_idx: Index of the current beat
            
        Returns:
            Dictionary of morphology features
        """
        features = {}
        
        # Get fiducial points for current beat
        r_peaks = fiducial_points['R_peaks']
        p_peaks = fiducial_points['P_peaks']
        q_peaks = fiducial_points['Q_peaks']
        s_peaks = fiducial_points['S_peaks']
        t_peaks = fiducial_points['T_peaks']
        
        # Current R-peak
        if beat_idx >= len(r_peaks):
            return None
            
        r_idx = r_peaks[beat_idx]
        
        # Find corresponding P, Q, S, T peaks for this beat
        p_idx = self._find_nearest_peak(p_peaks, r_idx, direction='before')
        q_idx = self._find_nearest_peak(q_peaks, r_idx, direction='before')
        s_idx = self._find_nearest_peak(s_peaks, r_idx, direction='after')
        t_idx = self._find_nearest_peak(t_peaks, r_idx, direction='after')
        
        # Get amplitudes
        features['R_amp'] = ecg_signal[r_idx] if r_idx is not None else np.nan
        features['P_amp'] = ecg_signal[p_idx] if p_idx is not None else np.nan
        features['Q_amp'] = ecg_signal[q_idx] if q_idx is not None else np.nan
        features['S_amp'] = ecg_signal[s_idx] if s_idx is not None else np.nan
        features['T_amp'] = ecg_signal[t_idx] if t_idx is not None else np.nan
        
        # Interval-based distances (in seconds)
        features['PR_interval'] = self._calc_interval(p_idx, r_idx)
        features['QR_interval'] = self._calc_interval(q_idx, r_idx)
        features['RS_interval'] = self._calc_interval(r_idx, s_idx)
        features['RT_interval'] = self._calc_interval(r_idx, t_idx)
        features['QS_interval'] = self._calc_interval(q_idx, s_idx)
        features['QT_interval'] = self._calc_interval(q_idx, t_idx)
        features['ST_interval'] = self._calc_interval(s_idx, t_idx)
        features['PS_interval'] = self._calc_interval(p_idx, s_idx)
        features['PT_interval'] = self._calc_interval(p_idx, t_idx)
        features['PQ_interval'] = self._calc_interval(p_idx, q_idx)
        
        # Euclidean distances
        features['PR_distance'] = self._calc_euclidean_distance(ecg_signal, p_idx, r_idx)
        features['QR_distance'] = self._calc_euclidean_distance(ecg_signal, q_idx, r_idx)
        features['RS_distance'] = self._calc_euclidean_distance(ecg_signal, r_idx, s_idx)
        features['RT_distance'] = self._calc_euclidean_distance(ecg_signal, r_idx, t_idx)
        features['QS_distance'] = self._calc_euclidean_distance(ecg_signal, q_idx, s_idx)
        features['QT_distance'] = self._calc_euclidean_distance(ecg_signal, q_idx, t_idx)
        features['ST_distance'] = self._calc_euclidean_distance(ecg_signal, s_idx, t_idx)
        features['PS_distance'] = self._calc_euclidean_distance(ecg_signal, p_idx, s_idx)
        features['PT_distance'] = self._calc_euclidean_distance(ecg_signal, p_idx, t_idx)
        
        # Slopes between peaks
        features['PR_slope'] = self._calc_slope(ecg_signal, p_idx, r_idx)
        features['QR_slope'] = self._calc_slope(ecg_signal, q_idx, r_idx)
        features['RS_slope'] = self._calc_slope(ecg_signal, r_idx, s_idx)
        features['RT_slope'] = self._calc_slope(ecg_signal, r_idx, t_idx)
        features['QS_slope'] = self._calc_slope(ecg_signal, q_idx, s_idx)
        features['QT_slope'] = self._calc_slope(ecg_signal, q_idx, t_idx)
        features['ST_slope'] = self._calc_slope(ecg_signal, s_idx, t_idx)
        features['PS_slope'] = self._calc_slope(ecg_signal, p_idx, s_idx)
        features['PT_slope'] = self._calc_slope(ecg_signal, p_idx, t_idx)
        
        # RR interval and HR
        if beat_idx < len(r_peaks) - 1:
            rr_interval = (r_peaks[beat_idx + 1] - r_peaks[beat_idx]) / self.sampling_rate
            features['RR'] = rr_interval
            features['HR'] = 60.0 / rr_interval if rr_interval > 0 else np.nan
        else:
            features['RR'] = np.nan
            features['HR'] = np.nan
        
        return features
    
    def _find_nearest_peak(self, peaks, ref_idx, direction='before', max_distance=None):
        """Find nearest peak to reference index"""
        if len(peaks) == 0 or ref_idx is None:
            return None
        
        peaks = np.array(peaks)
        peaks = peaks[~np.isnan(peaks)]
        
        if len(peaks) == 0:
            return None
        
        if direction == 'before':
            valid_peaks = peaks[peaks < ref_idx]
            if len(valid_peaks) == 0:
                return None
            return int(valid_peaks[-1])
        else:  # after
            valid_peaks = peaks[peaks > ref_idx]
            if len(valid_peaks) == 0:
                return None
            return int(valid_peaks[0])
    
    def _calc_interval(self, idx1, idx2):
        """Calculate time interval between two points in seconds"""
        if idx1 is None or idx2 is None:
            return np.nan
        return abs(idx2 - idx1) / self.sampling_rate
    
    def _calc_euclidean_distance(self, ecg_signal, idx1, idx2):
        """Calculate Euclidean distance between two points"""
        if idx1 is None or idx2 is None:
            return np.nan
        
        time_diff = abs(idx2 - idx1) / self.sampling_rate
        amp_diff = abs(ecg_signal[idx2] - ecg_signal[idx1])
        return np.sqrt(time_diff**2 + amp_diff**2)
    
    def _calc_slope(self, ecg_signal, idx1, idx2):
        """Calculate slope between two points"""
        if idx1 is None or idx2 is None or idx1 == idx2:
            return np.nan
        
        amp_diff = ecg_signal[idx2] - ecg_signal[idx1]
        time_diff = (idx2 - idx1) / self.sampling_rate
        return amp_diff / time_diff if time_diff != 0 else np.nan


class HRVFeatureExtractor:
    """Extract HRV time-domain features"""
    
    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate
    
    def extract_hrv_features(self, r_peaks, window_duration=60):
        """
        Extract 18 HRV time-domain features for a time window
        
        Args:
            r_peaks: Array of R-peak indices
            window_duration: Duration of window in seconds (default: 60s = 1 minute)
            
        Returns:
            Dictionary of HRV features
        """
        if len(r_peaks) < 2:
            return {}
        
        # Convert R-peaks to RR intervals (in ms)
        rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000
        
        # Use NeuroKit2 to compute HRV features
        hrv_features = nk.hrv_time(r_peaks, sampling_rate=self.sampling_rate)
        
        # Return as dictionary
        return hrv_features.to_dict(orient='records')[0] if len(hrv_features) > 0 else {}


class IntervalFeatureExtractor:
    """Extract interval-level features from beat-level predictions"""
    
    def __init__(self):
        pass
    
    def extract_interval_features(self, beat_predictions, hour_of_day=None):
        """
        Extract interval-level features from beat-level predictions
        
        Args:
            beat_predictions: Array of predicted probabilities for beats in interval
            hour_of_day: Hour of the day (0-23) for cyclical encoding
            
        Returns:
            Dictionary of interval features
        """
        features = {}
        
        if len(beat_predictions) == 0:
            return features
        
        # Percentage of beats classified as hypoglycemia (threshold = 0.5)
        features['pct_hypo_beats'] = np.mean(beat_predictions > 0.5)
        
        # Longest sequence of hypoglycemia beats
        binary_preds = (beat_predictions > 0.5).astype(int)
        features['longest_hypo_sequence'] = self._longest_sequence(binary_preds)
        
        # Mean predicted probability
        features['mean_prob'] = np.mean(beat_predictions)
        
        # Group probabilities into bins
        features['group1'] = np.mean((beat_predictions > 0.0) & (beat_predictions <= 0.2))
        features['group2'] = np.mean((beat_predictions > 0.2) & (beat_predictions <= 0.4))
        features['group3'] = np.mean((beat_predictions > 0.4) & (beat_predictions <= 0.6))
        features['group4'] = np.mean((beat_predictions > 0.6) & (beat_predictions <= 0.8))
        features['group5'] = np.mean((beat_predictions > 0.8) & (beat_predictions <= 1.0))
        
        # Cyclical encoding of hour
        if hour_of_day is not None:
            features['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
            features['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
        
        return features
    
    def _longest_sequence(self, binary_array):
        """Find longest consecutive sequence of 1s"""
        if len(binary_array) == 0:
            return 0
        
        max_length = 0
        current_length = 0
        
        for value in binary_array:
            if value == 1:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0
        
        return max_length
