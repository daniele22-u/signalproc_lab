"""
ECG Signal Processing Module
Handles preprocessing and fiducial point detection
"""

import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import neurokit2 as nk


class ECGProcessor:
    """Process ECG signals and detect fiducial points"""
    
    def __init__(self, sampling_rate=250):
        """
        Initialize ECG processor
        
        Args:
            sampling_rate: ECG sampling rate in Hz (default: 250Hz as per Zephyr Bioharness)
        """
        self.sampling_rate = sampling_rate
    
    def preprocess(self, ecg_signal):
        """
        Preprocess ECG signal: filtering and baseline correction
        
        Args:
            ecg_signal: Raw ECG signal array
            
        Returns:
            Cleaned ECG signal
        """
        # Clean ECG signal using NeuroKit2
        cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=self.sampling_rate)
        return cleaned_ecg
    
    def detect_fiducial_points(self, ecg_signal):
        """
        Detect R-peaks and fiducial points (P, Q, R, S, T)
        
        Args:
            ecg_signal: Cleaned ECG signal
            
        Returns:
            Dictionary containing indices of fiducial points for each beat
        """
        # Detect R-peaks
        peaks, info = nk.ecg_peaks(ecg_signal, sampling_rate=self.sampling_rate)
        r_peaks = info['ECG_R_Peaks']
        
        # Delineate ECG waves (P, Q, S, T peaks)
        signals, waves = nk.ecg_delineate(ecg_signal, r_peaks, sampling_rate=self.sampling_rate)
        
        fiducial_points = {
            'R_peaks': r_peaks,
            'P_peaks': waves.get('ECG_P_Peaks', []),
            'Q_peaks': waves.get('ECG_Q_Peaks', []),
            'S_peaks': waves.get('ECG_S_Peaks', []),
            'T_peaks': waves.get('ECG_T_Peaks', []),
            'P_onsets': waves.get('ECG_P_Onsets', []),
            'T_offsets': waves.get('ECG_T_Offsets', [])
        }
        
        return fiducial_points
    
    def segment_beats(self, ecg_signal, r_peaks, window_size=None):
        """
        Segment ECG signal into individual beats
        
        Args:
            ecg_signal: Cleaned ECG signal
            r_peaks: Indices of R-peaks
            window_size: Size of window around R-peak (default: 0.6s = 600ms worth of samples)
            
        Returns:
            List of beat segments
        """
        if window_size is None:
            # 600ms window (0.6 seconds * sampling_rate samples/second)
            window_size = int(0.6 * self.sampling_rate)
        
        beats = []
        half_window = window_size // 2
        
        for r_peak in r_peaks:
            start = max(0, r_peak - half_window)
            end = min(len(ecg_signal), r_peak + half_window)
            
            if end - start == window_size:
                beat = ecg_signal[start:end]
                beats.append(beat)
        
        return beats
