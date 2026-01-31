"""
Bowel Sound Analyzer Module
Main analysis orchestration and peak detection
"""

import numpy as np
import scipy.signal as signal
from typing import Dict, Tuple

from .preprocessing import AudioPreprocessor
from .features import FeatureExtractor


class BowelSoundAnalyzer:
    """
    Main analyzer for bowel sound signals
    """
    
    def __init__(self, audio: np.ndarray, sample_rate: int = 44100):
        """
        Initialize analyzer
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
        """
        self.audio = audio
        self.sample_rate = sample_rate
        self.duration = len(audio) / sample_rate
        
        self.preprocessor = AudioPreprocessor(sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate)
    
    def detect_events(
        self, 
        low_freq: float = 200, 
        high_freq: float = 800,
        threshold_factor: float = 0.6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect bowel sound events
        
        Args:
            low_freq: Low frequency cutoff
            high_freq: High frequency cutoff
            threshold_factor: Peak detection threshold (0-1)
            
        Returns:
            peaks: Array of peak indices
            envelope: Signal envelope
        """
        # Bandpass filter
        filtered = self.preprocessor.bandpass_filter(self.audio, low_freq, high_freq)
        
        # Extract envelope
        envelope = self.preprocessor.extract_envelope(filtered, smooth_window_ms=50)
        
        # Detect peaks
        threshold = threshold_factor * np.max(envelope)
        peaks, _ = signal.find_peaks(
            envelope,
            height=threshold,
            distance=int(0.1 * self.sample_rate),  # Min 100ms between peaks
            prominence=threshold * 0.3
        )
        
        return peaks, envelope
    
    def analyze(self) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Perform complete analysis
        
        Returns:
            analysis: Complete analysis dictionary
            peaks: Detected peak indices
            envelope: Signal envelope
        """
        # Detect events
        peaks, envelope = self.detect_events()
        
        # Extract all features
        temporal_features = self.feature_extractor.extract_temporal_features(peaks)
        variability_features = self.feature_extractor.extract_variability_features(peaks)
        energy_features = self.feature_extractor.extract_energy_features(self.audio)
        snr = self.feature_extractor.extract_snr(self.audio)
        band_features = self.feature_extractor.extract_frequency_band_features(self.audio)
        irregularity_features = self.feature_extractor.extract_irregularity_features(peaks)
        
        # Compile analysis
        analysis = {
            'file_info': {
                'fs': self.sample_rate,
                'duration_s': float(self.duration)
            },
            'events': temporal_features,
            'variability': variability_features,
            'energy': energy_features,
            'snr_db': snr,
            'frequency_band': band_features,
            'irregular_spacing': irregularity_features
        }
        
        return analysis, peaks, envelope